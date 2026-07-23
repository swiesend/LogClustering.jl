"""
    Anomaly.RRCF

Streaming random-cut / isolation forest anomaly detector — a
model-free, pure-Julia online signal for the rule engine.

A forest of isolation trees is built over a **reservoir sample** of
the stream (bounded memory). A new point is scored by its expected
isolation depth across the trees — the shorter the path needed to
isolate it, the more anomalous — normalised to `[0, 1]` via the
standard `c(n)` adjustment (Liu, Ting & Zhou 2008). Cuts are random
axis-parallel splits weighted by per-dimension span, i.e. the
random-cut-tree construction (Guha, Mishra, Roy & Schrijvers 2016).

This is the tractable, hot-path-friendly member of the random-cut
family: O(depth · n_trees) per score, reservoir-bounded, and the
forest is periodically rebuilt so it tracks the stream. (Full RRCF
*collusive displacement* needs incremental tree insert/delete; the
isolation-depth score is the shared foundation and is what the
literature validates for streaming anomaly detection.)

Score-then-learn: [`observe!`] scores a point against the model built
from prior data, *then* folds it into the reservoir — the correct
online ordering (a point is never scored against itself).

## Surface

- [`RCForest`] — the stateful detector.
- [`observe!`] — score one feature vector, then learn it. Returns the
  anomaly score in `[0, 1]` (`0.5` until the first forest is built).
- [`score`] — score without learning (read-only).
"""
module RRCF

using Random: AbstractRNG, MersenneTwister, rand

export RCForest, observe!, score

# ---------------------------------------------------------------------------
# Isolation-tree nodes.
# ---------------------------------------------------------------------------

# An internal node cuts dimension `dim` at `val`; a leaf holds the
# number of points that reached it (for the c(size) path adjustment).
struct ITNode
    is_leaf::Bool
    dim::Int
    val::Float64
    left::Int          # child indices into the tree's node vector (0 = none)
    right::Int
    size::Int          # leaf: points reaching here
    depth::Int
end

struct ITree
    nodes::Vector{ITNode}   # node 1 is the root
    height_limit::Int
end

# ---------------------------------------------------------------------------
# The forest.
# ---------------------------------------------------------------------------

"""
    RCForest(; dims, n_trees = 40, sample_size = 256,
             rebuild_every = 256, rng = MersenneTwister(0))

Streaming random-cut forest over `dims`-dimensional feature vectors.
`sample_size` reservoir points per rebuild; `n_trees` trees; the
forest is rebuilt every `rebuild_every` observations so it tracks the
stream.
"""
mutable struct RCForest
    dims::Int
    n_trees::Int
    sample_size::Int
    rebuild_every::Int
    rng::AbstractRNG
    reservoir::Vector{Vector{Float64}}
    n_seen::Int
    since_rebuild::Int
    trees::Vector{ITree}
    height_limit::Int
end

function RCForest(; dims::Integer,
                  n_trees::Integer = 40,
                  sample_size::Integer = 256,
                  rebuild_every::Integer = 256,
                  rng::AbstractRNG = MersenneTwister(0))
    dims >= 1 || throw(ArgumentError("dims must be ≥ 1"))
    hl = max(1, ceil(Int, log2(max(2, sample_size))))
    return RCForest(Int(dims), Int(n_trees), Int(sample_size),
                    Int(rebuild_every), rng, Vector{Float64}[], 0, 0,
                    ITree[], hl)
end

# Average path length of an unsuccessful BST search over n points —
# the isolation-forest normalisation constant.
@inline function _c(n::Int)
    n <= 1 && return 0.0
    return 2.0 * (log(n - 1) + 0.5772156649015329) - 2.0 * (n - 1) / n
end

# ---------------------------------------------------------------------------
# Tree construction (random cuts) over a subsample.
# ---------------------------------------------------------------------------

function _build_tree(pts::Vector{Vector{Float64}}, dims::Int,
                     height_limit::Int, rng::AbstractRNG)
    nodes = ITNode[]
    _grow!(nodes, pts, dims, 0, height_limit, rng)
    return ITree(nodes, height_limit)
end

# Recursively grow, returning this subtree's node index (1-based).
function _grow!(nodes::Vector{ITNode}, pts::Vector{Vector{Float64}},
                dims::Int, depth::Int, hlimit::Int, rng::AbstractRNG)
    n = length(pts)
    if n <= 1 || depth >= hlimit
        push!(nodes, ITNode(true, 0, 0.0, 0, 0, n, depth))
        return length(nodes)
    end
    # Random-cut: pick a dimension with nonzero span, cut uniformly in
    # its range. Weighting the dimension choice by span (as RRCF does)
    # is approximated by retrying dims until a non-degenerate one is
    # found; if none, this is a leaf (all points identical).
    dim = 0; lo = 0.0; hi = 0.0
    for _ in 1:min(dims, 8)
        d = rand(rng, 1:dims)
        mn = Inf; mx = -Inf
        @inbounds for p in pts
            v = p[d]
            v < mn && (mn = v)
            v > mx && (mx = v)
        end
        if mx > mn
            dim = d; lo = mn; hi = mx
            break
        end
    end
    if dim == 0
        push!(nodes, ITNode(true, 0, 0.0, 0, 0, n, depth))
        return length(nodes)
    end
    cut = lo + (hi - lo) * rand(rng)
    left = Vector{Float64}[]; right = Vector{Float64}[]
    @inbounds for p in pts
        if p[dim] < cut
            push!(left, p)
        else
            push!(right, p)
        end
    end
    # Reserve this node's slot, then fill children (post-order append).
    idx = length(nodes) + 1
    push!(nodes, ITNode(false, dim, cut, 0, 0, 0, depth))  # placeholder
    lidx = _grow!(nodes, left, dims, depth + 1, hlimit, rng)
    ridx = _grow!(nodes, right, dims, depth + 1, hlimit, rng)
    nodes[idx] = ITNode(false, dim, cut, lidx, ridx, 0, depth)
    return idx
end

# Path length of `x` through one tree (depth reached + c(leaf size)).
function _path_length(tree::ITree, x::AbstractVector{Float64})
    isempty(tree.nodes) && return 0.0
    i = 1
    @inbounds while true
        node = tree.nodes[i]
        if node.is_leaf
            return node.depth + _c(node.size)
        end
        i = x[node.dim] < node.val ? node.left : node.right
        i == 0 && return node.depth + 1.0
    end
end

# ---------------------------------------------------------------------------
# Scoring + learning.
# ---------------------------------------------------------------------------

"""
    score(f::RCForest, x) -> Float64

Anomaly score in `[0, 1]` for feature vector `x` without learning it.
`0.5` (neutral) before the first forest is built.
"""
function score(f::RCForest, x::AbstractVector{<:Real})
    xv = _as_f64(x, f.dims)                # validate dims first
    isempty(f.trees) && return 0.5
    total = 0.0
    @inbounds for t in f.trees
        total += _path_length(t, xv)
    end
    avg = total / length(f.trees)
    denom = _c(f.sample_size)
    denom <= 0 && return 0.5
    return clamp(2.0^(-avg / denom), 0.0, 1.0)
end

"""
    observe!(f::RCForest, x) -> Float64

Score `x` against the current forest, then fold it into the reservoir
(rebuilding the forest every `rebuild_every` observations). Returns
the anomaly score. Score-then-learn: `x` never influences its own
score.
"""
function observe!(f::RCForest, x::AbstractVector{<:Real})
    s = score(f, x)
    _reservoir_add!(f, _as_f64(x, f.dims))
    f.n_seen += 1
    f.since_rebuild += 1
    if (isempty(f.trees) && length(f.reservoir) >= min(f.sample_size, 8)) ||
       f.since_rebuild >= f.rebuild_every
        _rebuild!(f)
        f.since_rebuild = 0
    end
    return s
end

# Vitter reservoir sampling into a bounded buffer.
function _reservoir_add!(f::RCForest, xv::Vector{Float64})
    if length(f.reservoir) < f.sample_size
        push!(f.reservoir, xv)
    else
        j = rand(f.rng, 1:(f.n_seen + 1))
        j <= f.sample_size && (f.reservoir[j] = xv)
    end
    return nothing
end

function _rebuild!(f::RCForest)
    n = length(f.reservoir)
    n == 0 && return
    trees = Vector{ITree}(undef, f.n_trees)
    for t in 1:f.n_trees
        # Subsample (with replacement is fine for isolation trees).
        m = min(n, f.sample_size)
        sub = Vector{Vector{Float64}}(undef, m)
        @inbounds for k in 1:m
            sub[k] = f.reservoir[rand(f.rng, 1:n)]
        end
        trees[t] = _build_tree(sub, f.dims, f.height_limit, f.rng)
    end
    f.trees = trees
    return nothing
end

@inline function _as_f64(x::AbstractVector{<:Real}, dims::Int)
    length(x) == dims ||
        throw(ArgumentError("feature vector has $(length(x)) dims, expected $dims"))
    return x isa Vector{Float64} ? x : Float64.(x)
end

end # module RRCF
