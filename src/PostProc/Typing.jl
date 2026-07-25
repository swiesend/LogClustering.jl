"""
    PostProc.Typing

Slot typing — plan 001 Stage E′ post-processing step 2.

Given a cluster's template with raw `<*>` placeholders, infer the
*most-specific* typed placeholder (`<IP>`, `<INT>`, `<DECIMAL_EN>`, …)
by running the typed-regex battery from [`PreProc.Masking`] over each
slot's realisations. A slot earns a type only when at least
[`min_match_fraction`] of its realisations match the type's regex.

This complements [`PreProc.Masking`]: that one normalises *raw text*
before clustering; this one tightens *parsed templates* after
clustering. Together they keep the placeholder vocabulary stable
across the whole pipeline.
"""
module Typing

using ..Masking: DEFAULT_LABELS, DEFAULT_PATTERNS
using ..Purity: slot_realisations

export type_slot, type_template

# Compile each typed-slot regex once at module load — these are
# the same patterns used by `Masking.mask_lines`, but here we test
# *whole-string* match (anchored), not span search.
const _COMPILED_TYPED = [(label, Regex("^(?:" * pat * ")\$"))
                         for (label, pat) in zip(DEFAULT_LABELS, DEFAULT_PATTERNS)]

"""
    type_slot(values::AbstractVector{<:AbstractString};
              min_match_fraction = 0.95,
              wildcard = "<*>") -> String

Return the most-specific placeholder for the given slot realisations.
Types are tried in their `Masking.DEFAULT_LABELS` rank order
(specific first); the first one whose match fraction beats
`min_match_fraction` wins. If none does, return `wildcard`.

Empty input → `wildcard`.
"""
function type_slot(values::AbstractVector{<:AbstractString};
                   min_match_fraction::Real = 0.95,
                   wildcard::AbstractString = "<*>")
    isempty(values) && return String(wildcard)
    n = length(values)
    @inbounds for (label, re) in _COMPILED_TYPED
        hits = 0
        for v in values
            occursin(re, String(v)) && (hits += 1)
        end
        if hits / n >= min_match_fraction
            return "<$label>"
        end
    end
    return String(wildcard)
end

"""
    type_template(template, lines; wildcard = "<*>",
                  min_match_fraction = 0.95) -> String

Walk each placeholder slot in `template`; replace it with the
most-specific typed placeholder whose match fraction across
`lines` is at least `min_match_fraction`. Slots whose realisations
don't fit any type stay as `wildcard`.
"""
function type_template(template::AbstractString,
                       lines::AbstractVector{<:AbstractString};
                       wildcard::AbstractString = "<*>",
                       min_match_fraction::Real = 0.95,
                       placeholder_pattern::Regex = r"<\*>|<[A-Z][A-Z0-9_]*>")
    tpl_tokens = collect(split(template))
    slot_idxs = findall(t -> occursin(placeholder_pattern, t), tpl_tokens)
    isempty(slot_idxs) && return String(template)
    reals = slot_realisations(template, lines;
                              placeholder_pattern = placeholder_pattern)
    @inbounds for (k, idx) in enumerate(slot_idxs)
        tpl_tokens[idx] = type_slot(reals[k];
                                    min_match_fraction = min_match_fraction,
                                    wildcard = wildcard)
    end
    return join(tpl_tokens, ' ')
end

end # module Typing
