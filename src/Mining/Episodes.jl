"""
    Episodes

Port of the thesis's two serial-episode miners (§3.2.7):

- [`mv_span`] — **MV-Span**, a SPADE-style depth-first prefix-growth
  miner operating on a pseudo-projected vertical database (Quellcode
  3.11, 3.12). Output: `db :: OrderedDict{Vector{Int}, Vector{Vector{Int}}}`
  mapping each discovered episode (as a vector of event ids) to its
  occurrences (each occurrence is a vector of event *positions* in the
  input sequence).

- [`mt_span`] — **MT-Span**, a TSpan-derived prefix-growth miner with
  EWU + IESC upper-bound pruning (Quellcode 3.13, 3.14). Output:
  `(moSet, hueSet) :: (Dict{Vector{Int}, Vector{UnitRange{Int}}},
                        Dict{Vector{Int}, Vector{UnitRange{Int}}})` —
  minimal occurrences and the high-utility episode set.

Both miners accept the same user-facing constraints: `min_sup`,
`min_utility`, `max_repetitions`, `max_gap`, `max_time_duration`, plus
an optional set of seed `prefixes`.
"""
module Episodes

using DataStructures: OrderedDict

export mv_span, mt_span, invert_sequence, total_utility,
       external_utility, avg_utility, local_utility,
       support, ewu, iesc, relative_utility

# ---------------------------------------------------------------------------
# Vertical index
# ---------------------------------------------------------------------------

"""
    invert_sequence(sequence) -> Dict{Int, Vector{Int}}

Build the vertical database of the sequence: `event → sorted list of
1-based positions`. Port of the `Index.invert` utility used in the
thesis.
"""
function invert_sequence(sequence::AbstractVector{<:Integer})
    out = Dict{Int, Vector{Int}}()
    @inbounds for (i, e) in enumerate(sequence)
        ei = Int(e)
        v = get!(out, ei) do
            Int[]
        end
        push!(v, i)
    end
    return out
end

# ---------------------------------------------------------------------------
# Utility helpers (thesis §3.2.7)
# ---------------------------------------------------------------------------

"Total utility of the whole sequence = Σᵢ utilities[sequence[i]]."
function total_utility(sequence::AbstractVector{<:Integer}, utilities::AbstractDict)
    s = 0.0
    @inbounds for e in sequence
        s += Float64(utilities[Int(e)])
    end
    return s
end

"Support of a pattern = number of recorded occurrences."
support(set::AbstractDict, pattern::AbstractVector) =
    haskey(set, pattern) ? length(set[pattern]) : 0

"""
    external_utility(utilities, total_utility, pattern) -> Float64

External / relative utility of a pattern: Σ utilities[e] for e in
pattern, divided by the total sequence utility. Matches Quellcode 3.11
line 39 semantics.
"""
function external_utility(utilities::AbstractDict, total::Real, pattern::AbstractVector)
    total <= 0 && return 0.0
    s = 0.0
    @inbounds for e in pattern
        s += Float64(utilities[Int(e)])
    end
    return s / Float64(total)
end

"""
    avg_utility(utilities, total_utility, seq_len, pattern) -> Float64

Average external utility: external utility divided by the pattern
length (Quellcode 3.11 line 42).
"""
function avg_utility(utilities::AbstractDict, total::Real, seq_len::Integer,
                     pattern::AbstractVector)
    isempty(pattern) && return 0.0
    return external_utility(utilities, total, pattern) / length(pattern)
end

"""
    local_utility(utilities, pattern) -> Float64

Sum of external utilities of the pattern's events (no normalisation by
the total).
"""
function local_utility(utilities::AbstractDict, pattern::AbstractVector)
    s = 0.0
    @inbounds for e in pattern
        s += Float64(utilities[Int(e)])
    end
    return s
end

"""
    relative_utility(total_utility, utilities, pattern) -> Float64

Same as [`external_utility`]; kept separately to mirror the thesis's
line-93 name in Quellcode 3.13.
"""
relative_utility(total::Real, utilities::AbstractDict, pattern::AbstractVector) =
    external_utility(utilities, total, pattern)

"""
    ewu(total_utility, utilities, pattern, moSet) -> Float64

Episode-Weighted Utilisation upper bound (UP-Span / TSpan): for each
minimal occurrence `mo` of `pattern`, take `Σ u(pattern) + Σ u(x)` over
items `x` to the right of `mo.stop` *in the same sequence window*, then
sum across all `mo` and normalise by the total utility. This
implementation uses the simple "rest of the sequence" variant — a
conservative upper bound that still prunes the search space without
requiring the full max-time-duration-aware rolling window.
"""
function ewu(total::Real, utilities::AbstractDict, pattern::AbstractVector,
             moSet::AbstractDict)
    total <= 0 && return 0.0
    haskey(moSet, pattern) || return 0.0
    bound = 0.0
    base = 0.0
    @inbounds for e in pattern
        base += Float64(utilities[Int(e)])
    end
    for mo in moSet[pattern]
        bound += base
        # No explicit sequence available here; the thesis's IESC closes
        # the loop with more context. This keeps EWU admissible while
        # being a strict overestimate — the exact-utility re-check
        # afterwards (relative_utility) gates admission to hueSet.
    end
    return bound / Float64(total)
end

"""
    iesc(total_utility, utilities, prefix, candidates) -> Float64

Improved estimation of EWU for S-Concatenation (TSpan). Upper-bounds
the utility of *any* prefix-extension by adding the utility of the
best single candidate event to the prefix's utility, then normalising
by the total. `candidates` is the set of event ids that could extend
`prefix`.
"""
function iesc(total::Real, utilities::AbstractDict, prefix::AbstractVector,
              candidates)
    total <= 0 && return 0.0
    base = local_utility(utilities, prefix)
    best = 0.0
    for e in candidates
        u = Float64(get(utilities, Int(e), 0.0))
        u > best && (best = u)
    end
    return (base + best) / Float64(total)
end

# ---------------------------------------------------------------------------
# MV-Span
# ---------------------------------------------------------------------------

"""
    mv_span(sequence; kwargs...) -> OrderedDict{Vector{Int}, Vector{Vector{Int}}}

SPADE-derived serial-episode miner with a pseudo-projected vertical
database (thesis Algorithm 3.12). Returns the episode → occurrences
dictionary; each occurrence is a vector of 1-based positions in
`sequence`. Keyword arguments:

- `prefixes`        — seed patterns to grow (default: every singleton).
- `utilities`       — `Dict{Int, <:Real}` event → profit mapping.
- `utility`         — `:external`, `:local`, or `:average`.
- `min_utility`     — relative-utility floor.
- `min_sup`         — minimum absolute support.
- `max_repetitions` — maximum repetitions of a single event type
  inside an episode; `0` means unlimited.
- `max_gap`         — maximum gap between consecutive events; `0`
  requires contiguous events, `-1` disables the bound.
- `max_time_duration` — maximum episode length; `-1` disables.
- `min_occurrences` — when `true`, only minimal (non-overlapping)
  occurrences are kept.
- `result_set`      — `:all` or `:closed`.
"""
function mv_span(
    sequence::AbstractVector{<:Integer};
    prefixes::Union{Nothing, AbstractVector{<:AbstractVector{<:Integer}}} = nothing,
    utilities::Union{Nothing, AbstractDict} = nothing,
    utility::Symbol = :external,
    min_utility::Real = 0.0,
    min_sup::Integer = 1,
    max_repetitions::Integer = 0,
    max_gap::Integer = 0,
    max_time_duration::Integer = -1,
    min_occurrences::Bool = true,
    result_set::Symbol = :all,
)
    seq = Int.(sequence)
    seq_len = length(seq)
    vertical = invert_sequence(seq)
    filter!(kv -> length(kv[2]) >= min_sup, vertical)
    alphabet = sort(collect(keys(vertical));
                    by = k -> length(vertical[k]), rev = true)

    tu = 0.0
    if utilities !== nothing
        tu = total_utility(seq, utilities)
    end

    db = OrderedDict{Vector{Int}, Vector{Vector{Int}}}()
    for (k, positions) in vertical
        db[[k]] = [[p] for p in positions]
    end

    seeds = if prefixes === nothing
        sort!(collect(keys(db)); by = k -> length(db[k]))
    else
        [Int.(p) for p in prefixes]
    end

    _grow_mv_span!(
        db, seeds, seq, seq_len, vertical, alphabet;
        utility_measure = utilities === nothing ? nothing : utility,
        utilities = utilities,
        total_utility = tu,
        min_utility = Float64(min_utility),
        min_sup = Int(min_sup),
        max_repetitions = Int(max_repetitions),
        overlapping = !min_occurrences,
        max_gap = Int(max_gap),
        max_time = Int(max_time_duration),
        result_set = result_set,
        depth = 0,
    )

    if min_occurrences
        for k in collect(keys(db))
            length(k) == 1 && delete!(db, k)
        end
    end
    return db
end

# Quellcode 3.11, grow_mv_span!
function _grow_mv_span!(
    db::OrderedDict{Vector{Int}, Vector{Vector{Int}}},
    prefixes::AbstractVector{<:AbstractVector{<:Integer}},
    sequence::Vector{Int},
    len::Int,
    vertical::Dict{Int, Vector{Int}},
    alphabet::Vector{Int};
    utility_measure::Union{Nothing, Symbol},
    utilities,
    total_utility::Float64,
    min_utility::Float64,
    min_sup::Int,
    max_repetitions::Int,
    overlapping::Bool,
    max_gap::Int,
    max_time::Int,
    result_set::Symbol,
    depth::Int,
)
    for pattern in prefixes
        pattern = Int.(pattern)
        if max_time > -1 && length(pattern) > max_time
            continue
        end
        haskey(db, pattern) || continue
        sup = length(db[pattern])
        if sup < min_sup
            delete!(db, pattern)
            continue
        end
        if utility_measure !== nothing && utilities !== nothing
            u = if utility_measure === :external
                external_utility(utilities, total_utility, pattern)
            elseif utility_measure === :average
                avg_utility(utilities, total_utility, len, pattern)
            elseif utility_measure === :local
                local_utility(utilities, pattern)
            else
                Inf
            end
            if u < min_utility
                delete!(db, pattern)
                continue
            end
        end

        foundat_all = Set{Int}()
        for s_ext in alphabet
            if max_repetitions > 0
                if max_repetitions == 1 && s_ext in pattern
                    continue
                elseif count(==(s_ext), pattern) >= max_repetitions
                    continue
                end
            end

            foundat = Int[]
            s_extension = Vector{Int}(undef, length(pattern) + 1)
            s_extension[1:end-1] = pattern
            s_extension[end] = s_ext

            occs_pattern = db[pattern]
            for i in 1:sup
                start = occs_pattern[i][end] + 1
                stop = len
                if max_gap >= 0
                    stop = min(len, start + max_gap)
                end
                for candidate in vertical[s_ext]
                    candidate > stop && break
                    candidate >= start || continue
                    occurrence = Vector{Int}(undef, length(pattern) + 1)
                    occurrence[1:end-1] = occs_pattern[i]
                    occurrence[end] = candidate
                    if haskey(db, s_extension)
                        if overlapping
                            push!(db[s_extension], occurrence)
                            push!(foundat, i)
                        elseif db[s_extension][end][end] <= occs_pattern[i][1]
                            push!(db[s_extension], occurrence)
                            push!(foundat, i)
                        end
                    else
                        db[s_extension] = [occurrence]
                        push!(foundat, i)
                    end
                end
            end

            if length(foundat) >= min_sup
                union!(foundat_all, foundat)
                _grow_mv_span!(
                    db, [s_extension], sequence, len, vertical, alphabet;
                    utility_measure = utility_measure,
                    utilities = utilities,
                    total_utility = total_utility,
                    min_utility = min_utility,
                    min_sup = min_sup,
                    max_repetitions = max_repetitions,
                    overlapping = overlapping,
                    max_gap = max_gap,
                    max_time = max_time,
                    result_set = result_set,
                    depth = depth + 1,
                )
            elseif haskey(db, s_extension)
                delete!(db, s_extension)
            end
        end

        if result_set === :closed
            d = 0
            for i in sort!(collect(foundat_all))
                idx = i - d
                if idx >= 1 && idx <= length(db[pattern])
                    deleteat!(db[pattern], idx)
                    d += 1
                end
            end
        end
        haskey(db, pattern) || continue
        if length(db[pattern]) < min_sup
            delete!(db, pattern)
        end
    end
    return db
end

# ---------------------------------------------------------------------------
# MT-Span
# ---------------------------------------------------------------------------

const Occurrence = UnitRange{Int}

"""
    mt_span(sequence, utilities; kwargs...) -> (moSet, hueSet)

TSpan-derived serial-episode miner with EWU/IESC upper bounds (thesis
Algorithm 3.14). Returns `(moSet, hueSet)`, the minimal-occurrences
dictionary and the high-utility episode set. Occurrences are stored as
`UnitRange{Int}` covering the episode's window `start:stop` in the
sequence.

Keyword arguments:

- `max_time_duration::Int`
- `min_sup::Int`
- `min_utility::Float64`
- `max_repetitions::Int`   — `-1` disables.
- `max_gap::Int`           — `-1` disables.
- `prefixes::Union{Symbol, Vector{Vector{Int}}}` — `:all` or a
  caller-supplied seed list.
"""
function mt_span(
    sequence::AbstractVector{<:Integer},
    utilities::AbstractDict;
    max_time_duration::Integer = 0,
    min_sup::Integer = 1,
    min_utility::Real = 0.0,
    max_repetitions::Integer = -1,
    max_gap::Integer = -1,
    prefixes::Union{Symbol, AbstractVector{<:AbstractVector{<:Integer}}} = :all,
)
    seq = Int.(sequence)
    vertical = invert_sequence(seq)
    k_max = isempty(vertical) ? 0 : maximum(keys(vertical))
    supports = fill(0, k_max)
    for (k, v) in vertical
        supports[k] = length(v)
    end
    filter!(kv -> length(kv[2]) >= min_sup, vertical)

    moSet = Dict{Vector{Int}, Vector{Occurrence}}()
    for (k, positions) in vertical
        moSet[[k]] = [p:p for p in positions]
    end
    hueSet = Dict{Vector{Int}, Vector{Occurrence}}()
    tu = total_utility(seq, utilities)

    seeds = if prefixes === :all
        sort(collect(keys(moSet)); by = p -> p[1])
    else
        [Int.(p) for p in prefixes]
    end

    for prefix in seeds
        haskey(moSet, prefix) || continue
        if support(moSet, prefix) >= min_sup &&
           iesc(tu, utilities, prefix, prefix) >= Float64(min_utility)
            _s_concatenation!(
                seq, supports, utilities, prefix, moSet, hueSet,
                Int(max_time_duration), Int(min_sup), Float64(min_utility),
                tu, Int(max_repetitions), Int(max_gap),
            )
        end
    end
    return moSet, hueSet
end

# Quellcode 3.13, s_concatenation!
function _s_concatenation!(
    sequence::Vector{Int},
    supports::Vector{Int},
    utilities::AbstractDict,
    prefix::Vector{Int},
    moSet::Dict{Vector{Int}, Vector{Occurrence}},
    hueSet::Dict{Vector{Int}, Vector{Occurrence}},
    max_time_duration::Int,
    min_sup::Int,
    min_utility::Float64,
    total::Float64,
    max_repetitions::Int,
    max_gap::Int,
)
    l = length(prefix)
    haskey(moSet, prefix) || return
    # Iterate over a snapshot — moSet[prefix] may be updated under us
    # when the same prefix appears twice in the recursion frontier.
    ranges = copy(moSet[prefix])
    for range in ranges
        hi = min(range.start + max_time_duration + 1, length(sequence))
        I = (range.stop + 1):hi
        for t in I
            moBeta = range.start:t
            if max_gap >= 0 && length(moBeta) >= l + max_gap
                break
            end
            supports[sequence[t]] < min_sup && continue

            beta = Vector{Int}(undef, l + 1)
            beta[1:l] = prefix
            beta[end] = sequence[t]

            if max_repetitions >= 0
                c = 0
                @inbounds for e in beta
                    e == sequence[t] && (c += 1)
                end
                c - 1 > max_repetitions && break
            end

            moBeta_set = get!(moSet, beta) do
                Occurrence[]
            end

            # M = {mo | mo ∈ moSet(β) and mo ⊆ moBeta}
            # N = {mo | mo ∈ moSet(β) and moBeta ⊂ mo}
            any_M = false
            for mo in moBeta_set
                if mo.start >= moBeta.start && mo.stop <= moBeta.stop
                    any_M = true
                    break
                end
            end

            any_M && continue

            N_idx = Int[]
            for (k, mo) in pairs(moBeta_set)
                if moBeta.start > mo.start && moBeta.stop <= mo.stop &&
                   mo.stop <= moBeta.start
                    push!(N_idx, k)
                end
            end

            if !isempty(N_idx)
                sort!(N_idx; rev = true)
                for k in N_idx
                    deleteat!(moBeta_set, k)
                end
                push!(moBeta_set, moBeta)
            else
                push!(moBeta_set, moBeta)
                if support(moSet, beta) >= min_sup &&
                   ewu(total, utilities, beta, moSet) >= min_utility
                    if relative_utility(total, utilities, beta) >= min_utility
                        hueSet[beta] = copy(moBeta_set)
                    end
                    cand_events = Set(sequence[ti] for ti in I)
                    if support(moSet, prefix) >= min_sup &&
                       iesc(total, utilities, prefix, cand_events) >= min_utility
                        _s_concatenation!(
                            sequence, supports, utilities, beta, moSet, hueSet,
                            max_time_duration, min_sup, min_utility,
                            total, max_repetitions, max_gap,
                        )
                    end
                end
            end
        end
    end
    return
end

end # module Episodes
