"""
    Cluster.Pipeline

Standard clustering pipeline for plan 001 Stage E:

```
embeddings (D × N)  →  L2-normalise  →  [optional] dim-reduction  →  clusterer
                                         ├── umap   (PythonCall → umap-learn)
                                         └── ...
                                                                     ├── kmeans    (Clustering.jl, native)
                                                                     └── hdbscan   (PythonCall → hdbscan)
```

For "clustering-via-sparsity" (the KATE argmax alternative) see
[`LogClustering.Sparsity`].

Current implementation:
- [`l2_normalise`] — column-wise L2 normalisation, pure Julia.
- [`kmeans_cluster`] — wraps `Clustering.kmeans` (Clustering.jl).
- [`umap_reduce`] — calls `umap-learn` through PythonCall.
- [`hdbscan_cluster`] — calls `hdbscan` through PythonCall.

The two Python paths raise a clear, actionable error when the uv
virtualenv under `py/` isn't wired up; see `py/README.md`.
"""
module Pipeline

using Clustering: Clustering
using Distances: Distances, pairwise
using LinearAlgebra: norm
using Statistics

export l2_normalise, l2_normalise!, kmeans_cluster,
       umap_reduce, hdbscan_cluster, umap_hdbscan

# ---------------------------------------------------------------------------
# L2 normalise
# ---------------------------------------------------------------------------

"""
    l2_normalise(X::AbstractMatrix; ϵ = 1e-8) -> Matrix

Return a column-normalised copy of `X` — every column has unit `ℓ₂`
norm. Zero columns are left at zero (guarded by `ϵ`).
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
a convergence flag.
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
# PythonCall shared bootstrap
# ---------------------------------------------------------------------------

const _PY_MOD = Ref{Union{Nothing, Module}}(nothing)

function _require_pythoncall()
    _PY_MOD[] === nothing || return _PY_MOD[]
    mod = try
        Base.require(Base.PkgId(
            Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall"))
    catch err
        error("""
            PythonCall failed to load. This happens when the uv virtualenv
            under `py/` isn't reachable. Sync and re-launch Julia:

                cd py && uv sync && cd ..
                export JULIA_CONDAPKG_BACKEND=Null
                export JULIA_PYTHONCALL_EXE=\$(pwd)/py/.venv/bin/python
                julia --project

            Underlying error: $(sprint(showerror, err))
        """)
    end
    _PY_MOD[] = mod
    return mod
end

function _pyimport(py::Module, name::AbstractString, install_hint::AbstractString)
    try
        return py.pyimport(name)
    catch err
        error("""
            Could not import Python module `$name`. Install it into the
            uv virtualenv:

                $install_hint

            Underlying error: $(sprint(showerror, err))
        """)
    end
end

# ---------------------------------------------------------------------------
# UMAP via umap-learn
# ---------------------------------------------------------------------------

"""
    umap_reduce(X::AbstractMatrix; n_neighbors = 15, min_dist = 0.0,
                n_components = 15, metric = "euclidean",
                random_state = nothing) -> Matrix{Float64}

Reduce `X` (`features × samples`) to `n_components × samples` via
Python's `umap-learn` (McInnes & Healy 2018). Defaults match the plan
Stage E recipe (`n=15, min_dist=0.0`). Requires the uv virtualenv under
`py/` — see the module docstring for bootstrap steps.
"""
function umap_reduce(X::AbstractMatrix;
                     n_neighbors::Integer = 15,
                     min_dist::Real = 0.0,
                     n_components::Integer = 15,
                     metric::AbstractString = "euclidean",
                     random_state = nothing)
    n_neighbors >= 2 || throw(ArgumentError("n_neighbors must be ≥ 2"))
    n_components >= 1 || throw(ArgumentError("n_components must be ≥ 1"))
    size(X, 2) >= n_neighbors ||
        throw(ArgumentError("need at least n_neighbors=$(n_neighbors) samples; \
                             got $(size(X, 2))"))
    # PythonCall is loaded lazily via `Base.require`, so its methods live
    # in a newer world than this (precompiled) function. Load it FIRST
    # (so the require's world bump lands before the call below), then run
    # the Python work through `invokelatest` so those just-loaded methods
    # are visible — without this, the first UMAP call inside a single call
    # frame (e.g. the CLI's `main → cmd_rca → umap_reduce`) throws a
    # world-age MethodError.
    py = _require_pythoncall()
    return Base.invokelatest(_umap_reduce_impl, py, X, Int(n_neighbors),
                             Float64(min_dist), Int(n_components),
                             String(metric), random_state)
end

function _umap_reduce_impl(py, X, n_neighbors, min_dist, n_components,
                           metric, random_state)
    umap_mod = _pyimport(py, "umap", "cd py && uv add umap-learn && uv sync")
    kwargs = (n_neighbors = n_neighbors, min_dist = min_dist,
              n_components = n_components, metric = metric)
    reducer = random_state === nothing ?
        umap_mod.UMAP(; kwargs...) :
        umap_mod.UMAP(; kwargs..., random_state = Int(random_state))
    # umap-learn expects (samples, features); Julia is (features, samples).
    Xt = collect(permutedims(X))
    Y_py = reducer.fit_transform(Xt)
    Y = py.pyconvert(Matrix{Float64}, Y_py)
    return collect(permutedims(Y))
end

# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

"""
    hdbscan_cluster(X::AbstractMatrix; min_cluster_size = 10, metric = "euclidean")
        -> (assignments, outlier_scores)

Python's `hdbscan.HDBSCAN` through PythonCall. `X` is
`features × samples`. Returns 1-based cluster ids (noise stays `0`) and
the per-sample outlier scores.
"""
function hdbscan_cluster(X::AbstractMatrix;
                         min_cluster_size::Integer = 10,
                         metric::AbstractString = "euclidean")
    # See `umap_reduce`: load PythonCall first, then run the Python work
    # through `invokelatest` so the just-required methods are visible.
    py = _require_pythoncall()
    return Base.invokelatest(_hdbscan_cluster_impl, py, X,
                             Int(min_cluster_size), String(metric))
end

function _hdbscan_cluster_impl(py, X, min_cluster_size, metric)
    hdb = _pyimport(py, "hdbscan", "cd py && uv add hdbscan && uv sync")
    clusterer = hdb.HDBSCAN(
        min_cluster_size = min_cluster_size,
        metric = metric,
    )
    Xt = collect(permutedims(X))
    labels_py = clusterer.fit_predict(Xt)
    labels = py.pyconvert(Vector{Int}, labels_py) .+ 1
    # HDBSCAN uses -1 for noise; after +1 it becomes 0 — that's our noise tag.
    outlier = py.pyconvert(Vector{Float64}, clusterer.outlier_scores_)
    return (assignments = labels, outlier_scores = outlier)
end

# ---------------------------------------------------------------------------
# Convenience: full plan Stage E pipeline
# ---------------------------------------------------------------------------

"""
    umap_hdbscan(X; l2 = true,
                 n_neighbors = 15, min_dist = 0.0, n_components = 15,
                 min_cluster_size = 10) -> (assignments, embedding)

Plan 001 Stage E default: optional L2-normalise → UMAP → HDBSCAN.
Returns the HDBSCAN assignment vector and the UMAP `(n_components, samples)`
embedding so downstream code can plot or score it.
"""
function umap_hdbscan(X::AbstractMatrix;
                      l2::Bool = true,
                      n_neighbors::Integer = 15,
                      min_dist::Real = 0.0,
                      n_components::Integer = 15,
                      min_cluster_size::Integer = 10,
                      random_state = nothing)
    Xn = l2 ? l2_normalise(X) : X
    Y = umap_reduce(Xn;
                    n_neighbors = n_neighbors,
                    min_dist = min_dist,
                    n_components = n_components,
                    random_state = random_state)
    r = hdbscan_cluster(Y; min_cluster_size = min_cluster_size)
    return (assignments = r.assignments, embedding = Y)
end

end # module Pipeline
