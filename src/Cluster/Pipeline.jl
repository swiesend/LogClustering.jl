"""
    Cluster.Pipeline

Standard clustering pipeline for plan 001 Stage E:

```
embeddings (D × N)  →  L2-normalise  →  [optional] dim-reduction  →  clusterer
                                                                     ├── kmeans     (Clustering.jl, native)
                                                                     └── hdbscan    (PythonCall → hdbscan, lazy)
```

For "clustering-via-sparsity" (the KATE argmax alternative) see
[`LogClustering.Sparsity`].

Current implementation:
- [`l2_normalise`] — column-wise L2 normalisation, pure Julia.
- [`kmeans_cluster`] — wraps `Clustering.kmeans` (Clustering.jl). Uses
  k-means++ initialisation and returns a `(assignments, centers,
  converged)` `NamedTuple`.
- [`hdbscan_cluster`] — stub that calls Python `hdbscan` via
  `PythonCall`; raises a clear error when the uv virtualenv isn't
  wired up (see `py/README.md`). UMAP integration goes in here too
  once the Python env is present.
"""
module Pipeline

using Clustering: Clustering
using Distances: Distances, pairwise
using LinearAlgebra: norm
using Statistics

export l2_normalise, l2_normalise!, kmeans_cluster, hdbscan_cluster

# ---------------------------------------------------------------------------
# L2 normalise
# ---------------------------------------------------------------------------

"""
    l2_normalise(X::AbstractMatrix; ϵ = 1e-8) -> Matrix

Return a column-normalised copy of `X` — every column has unit `ℓ₂`
norm. Zero columns are left at zero (guarded by `ϵ` to avoid a divide
by zero).
"""
function l2_normalise(X::AbstractMatrix; ϵ::Real = 1e-8)
    out = similar(X, float(eltype(X)))
    copyto!(out, X)
    l2_normalise!(out; ϵ = ϵ)
    return out
end

function l2_normalise!(X::AbstractMatrix; ϵ::Real = 1e-8)
    @inbounds for j in axes(X, 2)
        n = zero(float(eltype(X)))
        for i in axes(X, 1)
            n += X[i, j] * X[i, j]
        end
        n = sqrt(n)
        n < ϵ && continue
        for i in axes(X, 1)
            X[i, j] /= n
        end
    end
    return X
end

# ---------------------------------------------------------------------------
# k-means via Clustering.jl
# ---------------------------------------------------------------------------

"""
    kmeans_cluster(X::AbstractMatrix, k::Integer;
                   maxiter = 100, display = :none) -> (assignments, centers, converged)

Thin wrapper over `Clustering.kmeans`. `X` is `(features, samples)`,
returns 1-based cluster ids, the `(features, k)` centroid matrix, and
a convergence flag. Uses k-means++ initialisation.
"""
function kmeans_cluster(X::AbstractMatrix, k::Integer;
                        maxiter::Integer = 100, display::Symbol = :none)
    k >= 1 || throw(ArgumentError("k must be ≥ 1"))
    size(X, 2) >= k ||
        throw(ArgumentError("need at least k samples; got $(size(X, 2)) for k=$k"))
    r = Clustering.kmeans(X, k; maxiter = maxiter, display = display)
    return (assignments = Clustering.assignments(r),
            centers     = r.centers,
            converged   = r.converged)
end

# ---------------------------------------------------------------------------
# HDBSCAN (lazy via PythonCall)
# ---------------------------------------------------------------------------

"""
    hdbscan_cluster(X::AbstractMatrix; min_cluster_size = 10, metric = "euclidean")
        -> (assignments, outlier_scores)

Run Python's `hdbscan.HDBSCAN` on `X` (features × samples) through
`PythonCall`. Assumes the uv virtualenv in `py/` has been synced
(see `py/README.md`) and the environment variables
`JULIA_CONDAPKG_BACKEND=Null` + `JULIA_PYTHONCALL_EXE=<py/.venv>` are
set before `using LogClustering`.

Returns 1-based cluster ids (noise stays as `0`) and the per-sample
outlier scores from HDBSCAN. Raises `ErrorException` with remediation
steps if Python isn't usable.
"""
function hdbscan_cluster(X::AbstractMatrix;
                         min_cluster_size::Integer = 10,
                         metric::AbstractString = "euclidean")
    py = try
        Base.require(Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
                                "PythonCall"))
    catch err
        error("PythonCall is not loaded. Install and configure the uv \
               virtualenv under `py/` (see py/README.md), then \
               `using LogClustering` from a fresh Julia session.")
    end
    hdbscan_mod = try
        py.pyimport("hdbscan")
    catch err
        error("""
            Could not import Python `hdbscan`. Check that:
              - `cd py && uv sync` has been run
              - JULIA_CONDAPKG_BACKEND=Null
              - JULIA_PYTHONCALL_EXE=$(abspath(joinpath(@__DIR__, "..", "..", "py", ".venv", "bin", "python")))
            Underlying error: $(sprint(showerror, err))
        """)
    end
    clusterer = hdbscan_mod.HDBSCAN(
        min_cluster_size = Int(min_cluster_size),
        metric = String(metric),
    )
    # HDBSCAN expects (samples, features); Julia is (features, samples).
    labels_py = clusterer.fit_predict(py.pyrowlist(collect(permutedims(X))))
    labels = Int.(py.pyconvert(Vector{Int}, labels_py)) .+ 1   # 0-based → 1-based, noise → 0
    outlier = py.pyconvert(Vector{Float64}, clusterer.outlier_scores_)
    return (assignments = labels, outlier_scores = outlier)
end

end # module Pipeline
