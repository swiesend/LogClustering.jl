"""
    Mining.Baselines

PrefixSpan, SPADE, and CM-SPADE adapted to **single-sequence frequent
serial-episode mining** — the same input shape as the thesis's
`mv_span` and `mt_span` so the four miners are directly comparable
on a log-event stream.

All three return the same shape of result: an `OrderedDict` from each
discovered episode (a `Vector{Int}` of event ids) to its list of
**non-overlapping minimal occurrences** (each occurrence is a
`Vector{Int}` of 1-based positions in the input sequence). Singletons
are filtered out by default — matching `mv_span`'s convention.

Why three? They all solve the same problem with different data
structures, which exposes different scaling regimes:

- [`prefixspan`] — Pei et al. 2001. Depth-first projected-database
  growth. Memory-light; fastest for *short* patterns over *long*
  alphabets.
- [`spade`] — Zaki 2001. Vertical id-list joins. Pre-computes one
  position list per item, then grows patterns by merging lists.
  Cheaper than re-projecting for *deep* patterns.
- [`cmspade`] — Fournier-Viger et al. 2014. SPADE plus a co-occurrence
  map (`CMAP`) that prunes candidate extensions before any join. Fastest
  on sparse alphabets where most item pairs *never* co-occur within
  `max_gap`.

Constraints supported (all keyword-only):

- `min_sup::Integer` — minimum support (count of non-overlapping
  occurrences). Required.
- `max_gap::Integer` — maximum gap between consecutive events;
  `-1` disables.
- `max_time_duration::Integer` — maximum pattern length in events;
  `-1` disables.
"""
module Baselines

using DataStructures: OrderedDict

export prefixspan, spade, cmspade

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

"Item → sorted positions (vertical-database style)."
function _vertical(seq::AbstractVector{<:Integer})
    out = Dict{Int, Vector{Int}}()
    @inbounds for (i, e) in enumerate(seq)
        push!(get!(out, Int(e), Int[]), i)
    end
    return out
end

"""
Greedy non-overlapping subset of `occurrences`, ordered by *end*
position. Two occurrences `o₁, o₂` overlap iff their position ranges
intersect; we keep the earliest-ending and discard any later
occurrence whose start ≤ that one's end.
"""
function _nonoverlapping(occurrences::AbstractVector{<:AbstractVector{<:Integer}})
    isempty(occurrences) && return Vector{Vector{Int}}()
    sorted = sort(collect(occurrences); by = o -> o[end])
    out = Vector{Vector{Int}}()
    push!(out, sorted[1])
    last_end = sorted[1][end]
    @inbounds for o in @view sorted[2:end]
        if o[1] > last_end
            push!(out, o)
            last_end = o[end]
        end
    end
    return out
end

@inline function _within_gap(prev_pos::Int, next_pos::Int, max_gap::Int)
    max_gap < 0 && return next_pos > prev_pos
    return next_pos > prev_pos && (next_pos - prev_pos) <= max_gap
end

# ---------------------------------------------------------------------------
# PrefixSpan — depth-first projected growth
# ---------------------------------------------------------------------------

"""
    prefixspan(sequence; min_sup, max_gap = -1, max_time_duration = -1)
        -> OrderedDict{Vector{Int}, Vector{Vector{Int}}}

Pei et al. 2001 prefix-projected sequential-pattern miner adapted to
serial episodes on one sequence. See the module docstring for output
shape and constraint semantics.
"""
function prefixspan(seq::AbstractVector{<:Integer};
                    min_sup::Integer,
                    max_gap::Integer = -1,
                    max_time_duration::Integer = -1)
    s = Int.(seq)
    n = length(s)
    vertical = _vertical(s)
    filter!(kv -> length(kv[2]) >= min_sup, vertical)
    result = OrderedDict{Vector{Int}, Vector{Vector{Int}}}()
    for (e, positions) in vertical
        seed = [Int[p] for p in positions]
        _ps_grow!(result, [e], seed, vertical, n,
                  Int(min_sup), Int(max_gap), Int(max_time_duration))
    end
    _drop_singletons!(result)
    return result
end

function _ps_grow!(result, pattern::Vector{Int},
                   occurrences::Vector{Vector{Int}},
                   vertical::Dict{Int, Vector{Int}},
                   len::Int,
                   min_sup::Int, max_gap::Int, max_time_duration::Int)
    minimal = _nonoverlapping(occurrences)
    length(minimal) < min_sup && return
    result[copy(pattern)] = minimal
    if max_time_duration >= 0 && length(pattern) > max_time_duration
        return
    end
    for (e_ext, positions) in vertical
        extended = Vector{Vector{Int}}()
        for occ in minimal
            last_pos = occ[end]
            stop = max_gap < 0 ? len : min(len, last_pos + max_gap)
            for p in positions
                p > stop && break
                p <= last_pos && continue
                # Minimal-occurrence semantics: take the FIRST valid
                # extension per source occurrence; subsequent ones
                # would be non-minimal continuations.
                push!(extended, vcat(occ, p))
                break
            end
        end
        if length(extended) >= min_sup
            push!(pattern, e_ext)
            _ps_grow!(result, pattern, extended, vertical, len,
                      min_sup, max_gap, max_time_duration)
            pop!(pattern)
        end
    end
end

# ---------------------------------------------------------------------------
# SPADE — vertical id-list joins
# ---------------------------------------------------------------------------

"""
    spade(sequence; min_sup, max_gap = -1, max_time_duration = -1)
        -> OrderedDict{Vector{Int}, Vector{Vector{Int}}}

Zaki 2001 vertical-list serial-episode miner. Same output shape as
`prefixspan`. The two should produce identical patterns under
identical constraints — they're alternative search strategies, not
different definitions of "frequent serial episode".
"""
function spade(seq::AbstractVector{<:Integer};
               min_sup::Integer,
               max_gap::Integer = -1,
               max_time_duration::Integer = -1)
    s = Int.(seq)
    n = length(s)
    vertical = _vertical(s)
    filter!(kv -> length(kv[2]) >= min_sup, vertical)
    result = OrderedDict{Vector{Int}, Vector{Vector{Int}}}()
    # Initial id-lists: every position is a length-1 occurrence.
    for (e, positions) in vertical
        seed = [Int[p] for p in positions]
        _spade_grow!(result, [e], seed, vertical, n,
                     Int(min_sup), Int(max_gap), Int(max_time_duration))
    end
    _drop_singletons!(result)
    return result
end

function _spade_grow!(result, pattern::Vector{Int},
                      idlist::Vector{Vector{Int}},
                      vertical::Dict{Int, Vector{Int}},
                      len::Int,
                      min_sup::Int, max_gap::Int, max_time_duration::Int)
    minimal = _nonoverlapping(idlist)
    length(minimal) < min_sup && return
    result[copy(pattern)] = minimal
    if max_time_duration >= 0 && length(pattern) > max_time_duration
        return
    end
    for (e_ext, positions_e) in vertical
        # Vertical join: for each minimal end-position of `pattern`,
        # find the next position of `e_ext` within max_gap.
        joined = Vector{Vector{Int}}()
        for occ in minimal
            last_pos = occ[end]
            for p in positions_e
                _within_gap(last_pos, p, max_gap) || (p > last_pos && break)
                if _within_gap(last_pos, p, max_gap)
                    push!(joined, vcat(occ, p))
                    break
                end
            end
        end
        if length(joined) >= min_sup
            push!(pattern, e_ext)
            _spade_grow!(result, pattern, joined, vertical, len,
                         min_sup, max_gap, max_time_duration)
            pop!(pattern)
        end
    end
end

# ---------------------------------------------------------------------------
# CM-SPADE — SPADE + co-occurrence-map pruning
# ---------------------------------------------------------------------------

"""
    cmspade(sequence; min_sup, max_gap = -1, max_time_duration = -1)
        -> OrderedDict{Vector{Int}, Vector{Vector{Int}}}

Fournier-Viger et al. 2014. Same as SPADE plus a co-occurrence map
(`CMAP`) that prunes candidate extensions before the join. CMAP is
keyed by the pattern's last event `a`; only candidates `b` that ever
appeared in the relation "`b` follows `a` within `max_gap` somewhere
in the sequence" are tried — sparse alphabets see a big constant-
factor speed-up over plain SPADE.
"""
function cmspade(seq::AbstractVector{<:Integer};
                 min_sup::Integer,
                 max_gap::Integer = -1,
                 max_time_duration::Integer = -1)
    s = Int.(seq)
    n = length(s)
    vertical = _vertical(s)
    filter!(kv -> length(kv[2]) >= min_sup, vertical)
    cmap = _build_cmap(s, keys(vertical), Int(max_gap))
    result = OrderedDict{Vector{Int}, Vector{Vector{Int}}}()
    for (e, positions) in vertical
        seed = [Int[p] for p in positions]
        _cmspade_grow!(result, [e], seed, vertical, cmap, n,
                       Int(min_sup), Int(max_gap), Int(max_time_duration))
    end
    _drop_singletons!(result)
    return result
end

"For each frequent item `a`, the set of items that ever follow it within `max_gap`."
function _build_cmap(seq::Vector{Int}, frequent_items, max_gap::Int)
    keep = Set(frequent_items)
    cmap = Dict{Int, Set{Int}}()
    n = length(seq)
    @inbounds for i in 1:n
        a = seq[i]
        a in keep || continue
        s = get!(cmap, a, Set{Int}())
        stop = max_gap < 0 ? n : min(n, i + max_gap)
        for j in (i + 1):stop
            b = seq[j]
            b in keep && push!(s, b)
        end
    end
    return cmap
end

function _cmspade_grow!(result, pattern::Vector{Int},
                        idlist::Vector{Vector{Int}},
                        vertical::Dict{Int, Vector{Int}},
                        cmap::Dict{Int, Set{Int}},
                        len::Int,
                        min_sup::Int, max_gap::Int, max_time_duration::Int)
    minimal = _nonoverlapping(idlist)
    length(minimal) < min_sup && return
    result[copy(pattern)] = minimal
    if max_time_duration >= 0 && length(pattern) > max_time_duration
        return
    end
    last_event = pattern[end]
    candidates = get(cmap, last_event, nothing)
    candidates === nothing && return
    for e_ext in candidates
        positions_e = get(vertical, e_ext, nothing)
        positions_e === nothing && continue
        joined = Vector{Vector{Int}}()
        for occ in minimal
            last_pos = occ[end]
            for p in positions_e
                _within_gap(last_pos, p, max_gap) || (p > last_pos && break)
                if _within_gap(last_pos, p, max_gap)
                    push!(joined, vcat(occ, p))
                    break
                end
            end
        end
        if length(joined) >= min_sup
            push!(pattern, e_ext)
            _cmspade_grow!(result, pattern, joined, vertical, cmap, len,
                           min_sup, max_gap, max_time_duration)
            pop!(pattern)
        end
    end
end

# ---------------------------------------------------------------------------
# Output canonicalisation
# ---------------------------------------------------------------------------

function _drop_singletons!(d::OrderedDict)
    for k in collect(keys(d))
        length(k) == 1 && delete!(d, k)
    end
    return d
end

end # module Baselines
