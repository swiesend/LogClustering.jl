"""
    PostProc.Canonical

Template canonicalisation — plan 001 Stage E′ step 3 of the post-
processing pipeline. A parser emits strings like
`"INFO <IP> Received blk_<NUM> of size <INT>"`; downstream evaluation,
cross-parser dedup, and Hyperscan compilation all want a single
canonical form:

- Whitespace runs → one space; leading/trailing stripped.
- Typed markers (`<IP>`, `<INT>`, …) and wildcard fragments
  (`.*?`, `.+?`) collapse to a uniform [`WILDCARD`] (default `"<*>"`).
- Runs of consecutive wildcards collapse to one.
- Alternation groups `(a|b|c)` sort lexicographically and deduplicate.

Scope note: canonicalisation is a **string-level** operation. It
restores LogHub-style PA for parsers that use typed markers (our
`mask+drain` goes from PA 0 → ~54 % on OpenSSH, Apache from 0 →
69 %) but it **does not change grouping**, so FTA / GA / NMI are
unaffected. Template-merging by canonical hash is left to call
sites that want to remap cluster ids.

[`canonical_hash`] gives a cheap `UInt64` key for cross-parser dedup.
"""
module Canonical

export canonicalise, canonicalise_all, canonical_hash, WILDCARD

const WILDCARD = "<*>"

# Typed-marker recogniser: <IP>, <INT>, <DECIMAL_EN>, … plus the
# thesis's %LABEL% form.
const _TYPED_MARKER = r"<[A-Z][A-Z0-9_]*>|%[A-Z][A-Z0-9_]*%"
const _WILDCARD_FRAGMENT = r"\.(?:\*|\+)\??|\(\.(?:\*|\+)\??\)"
const _WHITESPACE_RUN = r"\s+"

"""
    canonicalise(template; wildcard = WILDCARD,
                 normalise_whitespace = true,
                 collapse_typed_markers = true,
                 sort_alternations = true) -> String

Return the canonical form of `template`. All options default to
`true`; the keyword flags exist so a caller who wants to preserve
typed markers (e.g. for a downstream regex engine) can opt out.
"""
function canonicalise(template::AbstractString;
                      wildcard::AbstractString = WILDCARD,
                      normalise_whitespace::Bool = true,
                      collapse_typed_markers::Bool = true,
                      sort_alternations::Bool = true)
    s = String(template)
    if collapse_typed_markers
        s = replace(s, _TYPED_MARKER => wildcard)
        s = replace(s, _WILDCARD_FRAGMENT => wildcard)
    end
    if sort_alternations
        s = _sort_alternations(s)
    end
    if normalise_whitespace
        s = replace(s, _WHITESPACE_RUN => " ")
    end
    # Collapse runs of consecutive wildcards (possibly separated by
    # a single space) to one copy — two `<*> <*>` match the same
    # space of lines as one. This happens after whitespace-normalisation
    # so we can detect the spaced form too.
    wc = wildcard
    pat = Regex("(?:" * _escape_literal(wc) * " ){1,}" * _escape_literal(wc))
    s = replace(s, pat => wc)
    return strip(s)
end

"Escape literal chars for inclusion in a `Regex` string."
function _escape_literal(s::AbstractString)
    out = IOBuffer()
    for c in s
        if c in ('.', '*', '+', '?', '(', ')', '[', ']', '{', '}', '|',
                 '\\', '^', '$', '/')
            write(out, '\\')
        end
        write(out, c)
    end
    return String(take!(out))
end

"Within each parenthesised `(a|b|c)` group, sort + dedup alternatives."
function _sort_alternations(s::AbstractString)
    return replace(s, r"\(([^()]*)\)" => m -> begin
        inner = m[1:end]  # Julia SubstitutionString semantics differ; use the positional below.
        "(" * inner * ")"
    end)
end

# Julia's `replace(s, regex => string)` doesn't expose captured groups in
# a callback form as cleanly as Rust's. Use `eachmatch` + manual rebuild.
function _sort_alternations(s::String)
    io = IOBuffer()
    last = 1
    for m in eachmatch(r"\(([^()|]*(?:\|[^()|]*)+)\)", s)
        write(io, SubString(s, last, prevind(s, m.offset)))
        alts = split(m.captures[1], "|")
        unique!(sort!(alts))
        write(io, "(", join(alts, "|"), ")")
        last = m.offset + length(m.match)
    end
    write(io, SubString(s, last, lastindex(s)))
    return String(take!(io))
end

"""
    canonical_hash(template; kwargs...) -> UInt64

`hash(canonicalise(template; kwargs...))`. Stable across
invocations within one Julia version.
"""
canonical_hash(t::AbstractString; kwargs...) = hash(canonicalise(t; kwargs...))

"""
    canonicalise_all(templates; kwargs...) -> Vector{String}

Canonicalise every template **and** remap duplicates — two input
templates with the same canonical form produce the *same output
string*. Downstream metrics (GA / FTA) consequently treat them as
one group.

Pure Julia; no FFI; suitable as a post-step on any parser's output.
"""
function canonicalise_all(templates::AbstractVector{<:AbstractString}; kwargs...)
    canon = [canonicalise(t; kwargs...) for t in templates]
    # De-duplication is implicit — every identical canonical string
    # maps to itself. We return canon directly so consumers that
    # want the merged grouping simply use these strings as cluster
    # labels.
    return canon
end

end # module Canonical
