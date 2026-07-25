"""
    Parsers.Drain3

Julia port of Drain (He et al. 2017) / Drain3 — the de-facto deterministic
log parser baseline, fully offline, no Python dependency.

Drain is a streaming prefix-tree parser:

1. Tokenise each incoming line on whitespace.
2. Route the line through a fixed-depth search tree keyed by
   `(length, token₁, token₂, …, token_{depth-1})`. Tokens containing
   digits (or matching a caller-supplied *parametrise* predicate) are
   keyed as the wildcard `"<*>"` so variable parts don't explode the
   tree.
3. The leaf holds a short list of *log clusters*, each with a template.
   The line is compared against every cluster in its leaf by token
   similarity `identical_positions / length`. If the best beats
   `sim_th`, the matched cluster's template absorbs the new line
   (positions that disagree become `"<*>"`); otherwise a new cluster
   is created.

This implementation covers the 2017 paper's algorithm plus Drain3's
`depth`, `sim_th`, `max_children`, `max_clusters` knobs. Keeps per-line
state updates amortised O(children-per-node).

Usage:

```julia
d = Drain()                   # default: depth = 4, sim_th = 0.4
for line in lines
    cid, template = process!(d, line)
end
templates = parse_all(Drain(), lines)   # convenience one-shot form
```

`parse_all(d, lines)` runs `process!` for every line and returns the
per-line template vector, suitable for
[`LogClustering.Harness.run_parser`](@ref).
"""
module Drain3

export Drain, LogCluster, process!, parse_all, template_of,
       assign, id_sequence_matrix

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

"""A single discovered log template and the number of lines that matched it."""
mutable struct LogCluster
    id::Int
    template::Vector{String}
    size::Int
end

template_of(c::LogCluster) = join(c.template, ' ')

mutable struct TreeNode
    children::Dict{String, TreeNode}
    clusters::Vector{Int}           # indices into Drain.clusters
    TreeNode() = new(Dict{String, TreeNode}(), Int[])
end

"""
    Drain(; depth = 4, sim_th = 0.4, max_children = 100, max_clusters = 0,
          wildcard = "<*>", parametrize = default_parametrize)

A streaming Drain parser. See the module docstring for the algorithm.
`depth` includes the `length` routing step; `sim_th` is the matching
threshold; `max_children` caps per-node fan-out (overflow funnels into
`wildcard`); `max_clusters = 0` means unlimited.

`parametrize(token) -> Bool` decides which tokens are treated as
variable at tree-routing time. Default: any token containing an
ASCII digit.
"""
mutable struct Drain
    depth::Int                      # total parse-tree depth (incl. length node)
    sim_th::Float64
    max_children::Int
    max_clusters::Int               # 0 = unlimited
    wildcard::String
    parametrize::Function           # token → Bool; true → treat as variable
    root::TreeNode
    clusters::Vector{LogCluster}
    next_id::Int
    # --- LRU eviction bookkeeping (NOT persisted; rebuilt on load) ---
    # Parallel to `clusters`: `last_seen[i]` is the access tick when
    # `clusters[i]` was last touched; `owner[i]` is the leaf node whose
    # `clusters` index list references slot `i`. When `max_clusters` is
    # reached, the globally coldest cluster is evicted and its slot
    # reused — so `clusters` stays fixed-size (bounded memory) and no
    # index shifting is needed.
    access_tick::Int
    last_seen::Vector{Int}
    owner::Vector{TreeNode}
end

function Drain(; depth::Integer = 4,
               sim_th::Real = 0.4,
               max_children::Integer = 100,
               max_clusters::Integer = 0,
               wildcard::AbstractString = "<*>",
               parametrize = default_parametrize)
    depth >= 2 || throw(ArgumentError("depth must be ≥ 2"))
    0 <= sim_th <= 1 || throw(ArgumentError("sim_th must be in [0, 1]"))
    return Drain(Int(depth), Float64(sim_th), Int(max_children),
                 Int(max_clusters), String(wildcard),
                 parametrize, TreeNode(), LogCluster[], 1,
                 0, Int[], TreeNode[])
end

# Rebuild the parallel LRU arrays after a bundle load (the tree +
# clusters are deserialized, but the bookkeeping isn't persisted).
function rebuild_lru!(d::Drain)
    n = length(d.clusters)
    d.last_seen = zeros(Int, n)
    d.owner = Vector{TreeNode}(undef, n)
    d.access_tick = 0
    _index_nodes!(d, d.root)
    return d
end

function _index_nodes!(d::Drain, node::TreeNode)
    @inbounds for idx in node.clusters
        1 <= idx <= length(d.owner) && (d.owner[idx] = node)
    end
    for child in values(node.children)
        _index_nodes!(d, child)
    end
    return nothing
end

@inline _touch!(d::Drain, idx::Int) =
    (d.last_seen[idx] = (d.access_tick += 1); nothing)

"Any token with an ASCII digit is a parameter candidate."
function default_parametrize(t::AbstractString)
    @inbounds for c in t
        '0' <= c <= '9' && return true
    end
    return false
end

# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------

"""
    process!(d::Drain, line) -> (cluster_id::Int, template::String)

Route `line` through the parse tree, match against the leaf's
clusters, update the best match or create a new one. Returns the
cluster id and the (possibly freshly generalised) template string.
"""
function process!(d::Drain, line::AbstractString)
    tokens = String.(split(line))
    isempty(tokens) && return (0, "")

    node = _walk_down!(d, tokens)
    best_idx, best_sim = _best_match(d, node, tokens)

    if best_idx != 0 && best_sim >= d.sim_th
        slot = node.clusters[best_idx]
        cl = d.clusters[slot]
        _merge_into_template!(cl, tokens, d.wildcard)
        cl.size += 1
        _touch!(d, slot)
        return (cl.id, template_of(cl))
    end

    # New cluster. At the cap, evict the globally least-recently-used
    # cluster and reuse its slot (bounded memory, no index shifting) —
    # this replaces the old "stuff into the first leaf cluster" stub,
    # which both grew imprecise over time and threw when the current
    # leaf had no clusters yet.
    if d.max_clusters > 0 && length(d.clusters) >= d.max_clusters
        victim = _lru_victim(d)
        if victim != 0
            vnode = d.owner[victim]
            pos = findfirst(==(victim), vnode.clusters)
            pos === nothing || deleteat!(vnode.clusters, pos)
            cl = LogCluster(d.next_id, copy(tokens), 1)
            d.clusters[victim] = cl
            d.owner[victim] = node
            push!(node.clusters, victim)
            _touch!(d, victim)
            d.next_id += 1
            return (cl.id, template_of(cl))
        end
    end

    cl = LogCluster(d.next_id, copy(tokens), 1)
    push!(d.clusters, cl)
    push!(node.clusters, length(d.clusters))
    push!(d.last_seen, 0)
    push!(d.owner, node)
    _touch!(d, length(d.clusters))
    d.next_id += 1
    return (cl.id, template_of(cl))
end

# Index of the globally coldest cluster (min last_seen). O(n) scan —
# only runs when creating a new cluster past the cap, which is rare in
# steady state. Returns 0 when bookkeeping is empty (never at cap).
function _lru_victim(d::Drain)
    n = length(d.last_seen)
    n == 0 && return 0
    victim = 0
    best = typemax(Int)
    @inbounds for i in 1:n
        isassigned(d.owner, i) || continue
        if d.last_seen[i] < best
            best = d.last_seen[i]
            victim = i
        end
    end
    return victim
end

"""
    parse_all(d::Drain, lines) -> Vector{String}

Stream every line through `process!`, then remap each line to its
*final* cluster template (after every subsequent merge). This matches
the LogHub-2.0 evaluation convention — the inferred template for
line `i` is the cluster template as it stands at the end of the run,
not the template that existed when line `i` was first seen.

Starts from the current state — pass a freshly constructed `Drain`
to run in isolation.
"""
function parse_all(d::Drain, lines::AbstractVector{<:AbstractString})
    cluster_ids = Vector{Int}(undef, length(lines))
    @inbounds for (i, l) in enumerate(lines)
        cid, _ = process!(d, String(l))
        cluster_ids[i] = cid
    end
    by_id = Dict(cl.id => template_of(cl) for cl in d.clusters)
    out = Vector{String}(undef, length(lines))
    @inbounds for i in eachindex(cluster_ids)
        cid = cluster_ids[i]
        out[i] = get(by_id, cid, "")
    end
    return out
end

# ---------------------------------------------------------------------------
# Read-only classification + template-id sequence featurisation (DeepLog).
# ---------------------------------------------------------------------------

"""
    assign(d::Drain, line) -> Int

Read-only classification: route `line` through the *existing* parse
tree and return the id of the best-matching cluster (similarity ≥
`sim_th`), or `0` when the line reaches no leaf or matches nothing.
Unlike [`process!`], this never creates clusters, generalises a
template, or touches recency — for scoring new lines against a frozen
model (e.g. a template-id sequence model at inference).
"""
function assign(d::Drain, line::AbstractString)
    tokens = String.(split(line))
    isempty(tokens) && return 0
    node = _walk_down(d, tokens)
    node === nothing && return 0
    best_idx, best_sim = _best_match(d, node, tokens)
    (best_idx != 0 && best_sim >= d.sim_th) || return 0
    return d.clusters[node.clusters[best_idx]].id
end

# Read-only tree walk: follow existing children only, falling back to
# the wildcard sibling; return the deepest reachable node or `nothing`.
function _walk_down(d::Drain, tokens::Vector{String})
    node = get(d.root.children, string(length(tokens)), nothing)
    node === nothing && return nothing
    max_token_layers = max(0, d.depth - 2)
    @inbounds for i in 1:min(max_token_layers, length(tokens))
        tok = tokens[i]
        key = d.parametrize(tok) ? d.wildcard : tok
        nxt = get(node.children, key, nothing)
        if nxt === nothing
            nxt = get(node.children, d.wildcard, nothing)
            nxt === nothing && return node
        end
        node = nxt
    end
    return node
end

"""
    id_sequence_matrix(d::Drain, lines; seqlen = 16, pad_id = 1,
                       learn = false) -> Matrix{Int}

DeepLog-style featurisation: map each line to its Drain template id,
then window the *sequence of template ids across consecutive lines*
into a `(seqlen, n_lines)` matrix. Column `j` holds the `seqlen`
template ids ending at line `j` (left-padded with `pad_id`), so the
model sees the recent log-key history predicting the next key.

Template ids are shifted by `+1` so cluster id 1 doesn't collide with
`pad_id = 1`; an empty line or an unmatched line (`learn = false`)
maps to `pad_id`. With `learn = true` the Drain is *grown* (via
[`process!`]) — used at training; with `learn = false` it is read
only (via [`assign`]) — used at inference so ids never exceed the
trained vocabulary.
"""
function id_sequence_matrix(d::Drain,
                            lines::AbstractVector{<:AbstractString};
                            seqlen::Integer = 16,
                            pad_id::Integer = 1,
                            learn::Bool = false)
    seqlen >= 1 || throw(ArgumentError("seqlen must be ≥ 1"))
    n = length(lines)
    ids = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        cid = learn ? process!(d, String(lines[i]))[1] : assign(d, String(lines[i]))
        ids[i] = cid <= 0 ? Int(pad_id) : cid + 1
    end
    X = fill(Int(pad_id), Int(seqlen), n)
    @inbounds for j in 1:n
        k = min(j, Int(seqlen))
        for i in 1:k
            X[Int(seqlen) - k + i, j] = ids[j - k + i]
        end
    end
    return X
end

# ---------------------------------------------------------------------------
# Tree routing
# ---------------------------------------------------------------------------

# Depth is counted inclusive of the leaf: d.depth = 4 means:
#   layer 0: root
#   layer 1: keyed by length
#   layer 2: keyed by tokens[1]
#   layer 3: keyed by tokens[2]  (leaf)
#
# If the line has fewer tokens than `d.depth - 1` we descend as far as
# possible; the node returned is a leaf node (or the deepest we can
# reach), where we keep the cluster list.
function _walk_down!(d::Drain, tokens::Vector{String})
    key = string(length(tokens))
    node = _child_or_create!(d, d.root, key)
    # Drop in at most `d.depth - 2` token-keyed layers (layer 1 = length).
    max_token_layers = max(0, d.depth - 2)
    @inbounds for i in 1:min(max_token_layers, length(tokens))
        tok = tokens[i]
        key = d.parametrize(tok) ? d.wildcard : tok
        node = _child_or_create!(d, node, key)
    end
    return node
end

function _child_or_create!(d::Drain, parent::TreeNode, key::AbstractString)
    c = get(parent.children, String(key), nothing)
    if c !== nothing
        return c
    end
    # Capacity: if we can still add a literal child, do so; otherwise
    # funnel into the wildcard sibling (creating it if needed).
    if length(parent.children) < d.max_children || key == d.wildcard
        child = TreeNode()
        parent.children[String(key)] = child
        return child
    end
    wc = get(parent.children, d.wildcard, nothing)
    if wc === nothing
        wc = TreeNode()
        parent.children[d.wildcard] = wc
    end
    return wc
end

# ---------------------------------------------------------------------------
# Similarity + template update
# ---------------------------------------------------------------------------

"Return (index into node.clusters, similarity) of the best leaf match; `(0, 0.0)` if none."
function _best_match(d::Drain, node::TreeNode, tokens::Vector{String})
    best_sim = 0.0
    best_i = 0
    @inbounds for (i, cid) in enumerate(node.clusters)
        cl = d.clusters[cid]
        length(cl.template) == length(tokens) || continue
        sim = _similarity(cl.template, tokens, d.wildcard)
        if sim > best_sim
            best_sim = sim
            best_i = i
        end
    end
    return best_i, best_sim
end

function _similarity(a::Vector{String}, b::Vector{String}, wc::AbstractString)
    n = length(a)
    n == 0 && return 0.0
    match = 0
    @inbounds for i in 1:n
        if a[i] == wc
            match += 1
        elseif a[i] == b[i]
            match += 1
        end
    end
    return match / n
end

function _merge_into_template!(cl::LogCluster, tokens::Vector{String}, wc::String)
    @inbounds for i in eachindex(cl.template)
        if cl.template[i] != wc && cl.template[i] != tokens[i]
            cl.template[i] = wc
        end
    end
end

end # module Drain
