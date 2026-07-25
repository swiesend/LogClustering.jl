"""
    PostProc.Purity

Slot-purity re-check — plan 001 Stage E′ post-processing step 1.

A parser emits a template like `INFO request <NUM>` because the
`<NUM>` slot was *meant* to hold variable values. But sometimes
every line in a cluster has the same value at that slot — the slot
should have been a literal. [`slot_entropy`] measures how variable
each slot's realisations actually are; [`promote_pure_slots`]
rewrites the template to fold low-entropy slots back to their
single observed value.

The threshold is in *bits*: the default `τ = 0.2` matches the
plan's recommendation. A slot whose empirical entropy is below `τ`
is treated as functionally constant.
"""
module Purity

using Statistics

export slot_entropy, promote_pure_slots, slot_realisations

"""
    slot_realisations(template, lines; placeholder_pattern = r"<\\*>|<[A-Z][A-Z0-9_]*>")
        -> Vector{Vector{String}}

For each placeholder slot in `template`, return the list of strings
it took on across the matching `lines`. Lines whose token count
doesn't agree with `template`'s contribute the empty string at every
slot (caller-side defensive default).

Output is one `Vector{String}` per slot, in left-to-right order.
"""
function slot_realisations(template::AbstractString,
                           lines::AbstractVector{<:AbstractString};
                           placeholder_pattern::Regex = r"<\*>|<[A-Z][A-Z0-9_]*>")
    tpl_tokens = split(template)
    slot_idxs = findall(t -> occursin(placeholder_pattern, t), tpl_tokens)
    realisations = [String[] for _ in slot_idxs]
    @inbounds for line in lines
        toks = split(line)
        length(toks) == length(tpl_tokens) || continue
        for (i, slot) in enumerate(slot_idxs)
            push!(realisations[i], String(toks[slot]))
        end
    end
    return realisations
end

"""
    slot_entropy(vals::AbstractVector{<:AbstractString}) -> Float64

Shannon entropy in bits of the empirical distribution over `vals`.
Constant input → 0; uniform N values → log2(N).
"""
function slot_entropy(vals::AbstractVector{<:AbstractString})
    isempty(vals) && return 0.0
    counts = Dict{String, Int}()
    for v in vals
        counts[v] = get(counts, v, 0) + 1
    end
    n = length(vals)
    h = 0.0
    @inbounds for c in Base.values(counts)
        p = c / n
        h -= p * log2(p)
    end
    return h
end

"""
    promote_pure_slots(template, lines; τ = 0.2,
                       placeholder_pattern = r"<\\*>|<[A-Z][A-Z0-9_]*>")
        -> String

Walk every placeholder slot in `template`; if its realisations
across `lines` have entropy < `τ`, replace the placeholder with the
sole / most-common value. Returns the updated template.
"""
function promote_pure_slots(template::AbstractString,
                            lines::AbstractVector{<:AbstractString};
                            τ::Real = 0.2,
                            placeholder_pattern::Regex = r"<\*>|<[A-Z][A-Z0-9_]*>")
    tpl_tokens = collect(split(template))
    slot_idxs = findall(t -> occursin(placeholder_pattern, t), tpl_tokens)
    isempty(slot_idxs) && return String(template)
    reals = slot_realisations(template, lines;
                              placeholder_pattern = placeholder_pattern)
    @inbounds for (k, idx) in enumerate(slot_idxs)
        vals = reals[k]
        isempty(vals) && continue
        if slot_entropy(vals) < τ
            tpl_tokens[idx] = _mode(vals)
        end
    end
    return join(tpl_tokens, ' ')
end

"Most-frequent value, tie-broken by first-seen order."
function _mode(values::AbstractVector{<:AbstractString})
    counts = Dict{String, Int}()
    order  = String[]
    @inbounds for v in values
        if !haskey(counts, v)
            push!(order, String(v))
        end
        counts[String(v)] = get(counts, String(v), 0) + 1
    end
    best = order[1]; bestn = counts[best]
    @inbounds for v in @view order[2:end]
        if counts[v] > bestn
            best = v; bestn = counts[v]
        end
    end
    return best
end

end # module Purity
