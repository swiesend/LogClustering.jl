"""
    Cluster.Sparsity

Clustering via KATE's competitive bottleneck — the thesis's native
interpretability signal (§3.2.4). Every input passed through a
`KCompetetive` layer has only `k` neurons active; the set of active
neurons (by index, and optionally by sign) forms a discrete *sparsity
label*. Inputs with identical active sets land in the same cluster
without ever computing a distance.

This is the "sparsity-cluster" leg of plan 001 Stage E, the baseline
against which UMAP + HDBSCAN is compared to isolate whether KATE's
bottleneck *is* its own clusterer.
"""
module Sparsity

using Statistics

export sparsity_labels, sparsity_clusters, top_k_indices

# ---------------------------------------------------------------------------
# Core: top-k active-neuron tuple per column
# ---------------------------------------------------------------------------

"""
    top_k_indices(z::AbstractVector, k::Integer; signed = false) -> NTuple

Indices of the `k` largest-magnitude entries of `z`, sorted ascending.
When `signed = true`, each index is paired with the sign of `z[i]`
(`+1` or `-1`) so `(1, +1)` and `(1, -1)` are *different* labels — this
mirrors KATE's split of positive vs. negative winners.
"""
function top_k_indices(z::AbstractVector, k::Integer; signed::Bool = false)
    k >= 1 || throw(ArgumentError("k must be ≥ 1"))
    n = length(z)
    kk = min(k, n)
    # partialsortperm by |z| descending, then sort the chosen indices
    # ascending so every k-tuple has a canonical orientation.
    idx = partialsortperm(z, 1:kk; by = abs, rev = true)
    sort!(idx)
    if signed
        return Tuple((i, z[i] >= 0 ? Int8(1) : Int8(-1)) for i in idx)
    else
        return Tuple(idx)
    end
end

"""
    sparsity_labels(Z::AbstractMatrix, k::Integer; signed = false) -> Vector

One label per column of `Z`, extracted by [`top_k_indices`]. `Z` is the
output of a `KCompetetive` / DeepKATE encoder run in test mode — shape
`(latent_dim, batch)`.
"""
function sparsity_labels(Z::AbstractMatrix, k::Integer; signed::Bool = false)
    [top_k_indices(view(Z, :, j), k; signed = signed) for j in axes(Z, 2)]
end

"""
    sparsity_clusters(Z::AbstractMatrix, k::Integer; signed = false)
        -> (assignments :: Vector{Int}, label_of_cluster :: Vector)

Assign each column of `Z` to an integer cluster id (1-based, stable
first-seen order), and return the list of sparsity labels corresponding
to each cluster.
"""
function sparsity_clusters(Z::AbstractMatrix, k::Integer; signed::Bool = false)
    raw = sparsity_labels(Z, k; signed = signed)
    by_label = Dict{eltype(raw), Int}()
    assign = Vector{Int}(undef, length(raw))
    label_of = Vector{eltype(raw)}()
    @inbounds for (j, lab) in enumerate(raw)
        id = get(by_label, lab, 0)
        if id == 0
            push!(label_of, lab)
            id = length(label_of)
            by_label[lab] = id
        end
        assign[j] = id
    end
    return assign, label_of
end

end # module Sparsity
