"""
    PreProc.Masking

Typed-slot masking — the thesis's "Parameterisierte Variablen" and plan
001 Stage E′ step 3. Replaces every match of a ranked battery of
typed regexes with a `<TYPE>` placeholder, so downstream parsers,
embedders and metrics see a structurally clean message.

The matcher is `LogClustering.Rust.parse_line` (the Rust
regex-cascade from thesis Algorithm 3.1) — so this module is a thin
Julia façade that:

1. Carries a sensible default ranking of typed regexes
   ([`DEFAULT_LABELS`], [`DEFAULT_PATTERNS`]).
2. Exposes [`mask_line`] / [`mask_lines`] — run the ranking over a
   string or vector of strings and emit `<LABEL>` wherever the
   ranking matched, leaving raw runs untouched.

Ranking follows the Stage E′ rule *complex patterns first* so they
win over generic ones (e.g. the ISO-8601 timestamp matches before
generic `\\d+`; IP wins over bare numbers). Callers can pass their
own `labels` / `patterns` to [`mask_line`] to extend or reorder.
"""
module Masking

using ..Rust: Rust

export DEFAULT_LABELS, DEFAULT_PATTERNS, SlotValue,
       mask_line, mask_lines, mask_line_with_values, mask_lines_with_values

# ---------------------------------------------------------------------------
# Default typed ranking
# ---------------------------------------------------------------------------
#
# One-entry-per-line mirrors the plan's Stage E′ step 3 catalogue.
# Order matters: the Rust cascade finds all non-overlapping matches
# of pattern[1] first, recurses on the unmatched substrings with
# pattern[2…], etc. Tie-breaking: longer / more specific shapes are
# ranked earlier.

const DEFAULT_LABELS = [
    "TIMESTAMP",    # ISO-8601 / RFC-3339
    "IPV6",
    "IP",
    "MAC",
    "UUID",
    "SHA",          # 40- or 64-hex-char checksum
    "HEX",          # 0x-prefixed hex address
    "URL",
    "EMAIL",
    "PATH",         # unix path
    "QUOTED",       # "…" double-quoted string
    "DURATION",     # 42ms / 13.5s / 1h / 2m
    "SIZE",         # 12KB / 4.5GiB
    "DECIMAL_EN",   # 1,234.56  — dot decimal, optional comma thousands
    "DECIMAL_DE",   # 1.234,56  — comma decimal, optional dot thousands
    "INT",          # 42 / 1,234 / 1.234 — no fractional part
]

const DEFAULT_PATTERNS = [
    raw"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b",
    raw"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b",
    raw"\b\d{1,3}(?:\.\d{1,3}){3}\b",
    raw"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b",
    raw"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
    raw"\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b",
    raw"\b0x[0-9a-fA-F]+\b",
    raw"https?://[^\s\"<>]+",
    raw"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    raw"(?:^|[\s(\[=])(/[^\s\"<>'\]\)\)]*)",
    raw"\"[^\"]*\"",
    raw"\b\d+(?:[.,]\d+)?(?:ms|us|ns|μs|s|m|h|d)\b",
    raw"\b\d+(?:[.,]\d+)?(?:[KMGTP]i?B|[kmgtp]b)\b",
    # DECIMAL_EN: dot decimal, optional comma thousands. To avoid
    # eating `1.234` (which is a DE-thousands-grouped integer, not an
    # EN decimal `1.234`), the fractional part must be 1–2 or 4+
    # digits — i.e. *not* exactly three, the length that collides
    # with a plain thousands group. The leading `-?\b` and trailing
    # `\b` stop the regex from chewing partial matches and leaving
    # digit crumbs behind. Unambiguous shapes still match: `3.14`,
    # `-0.5`, `1,234.56`, `1,234,567.8901`.
    raw"-?\b(?:\d{1,3}(?:,\d{3})+|\d+)\.(?:\d{1,2}|\d{4,})\b",
    # DECIMAL_DE: mirror (dot thousands, comma decimal).
    raw"-?\b(?:\d{1,3}(?:\.\d{3})+|\d+),(?:\d{1,2}|\d{4,})\b",
    # INT: plain or thousands-grouped integer. Ranked *after* the two
    # decimal slots so the ambiguous `1.234` / `1,234` shapes settle
    # here as integers rather than mis-splitting.
    raw"-?\b(?:\d{1,3}(?:,\d{3})+|\d{1,3}(?:\.\d{3})+|\d+)\b",
]

@assert length(DEFAULT_LABELS) == length(DEFAULT_PATTERNS)

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

"""
    SlotValue(label, value, start, stop)

One captured value from [`mask_line_with_values`] — the `label` (e.g.
`"IP"`), the raw matched substring `value`, and the 1-based inclusive
byte range `(start, stop)` in the original line. Mirrors the thesis's
`LogAttr` record (Quellcode 3.5) minus the occurrence-interval wrapper.
"""
struct SlotValue
    label::String
    value::String
    start::Int
    stop::Int
end

"""
    mask_line(line; labels = DEFAULT_LABELS, patterns = DEFAULT_PATTERNS,
              template = "<\$LABEL>") -> String

Run the ranked regex battery over `line`; substitute every labelled
span with the template (default `"<LABEL>"`). Raw runs between
matches survive verbatim.

`template` is a `printf`-like string with a single `\$LABEL`
placeholder. Use `"<\$LABEL>"` for LogHub-style output or
`"%\$LABEL%"` for the thesis's descriptive-log-key form.
"""
function mask_line(line::AbstractString;
                   labels::AbstractVector{<:AbstractString} = DEFAULT_LABELS,
                   patterns::AbstractVector{<:AbstractString} = DEFAULT_PATTERNS,
                   template::AbstractString = "<\$LABEL>")
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))
    spans = Rust.parse_line(line, labels, patterns)
    io = IOBuffer()
    @inbounds for s in spans
        if s.label === nothing
            write(io, SubString(line, s.start, s.stop))
        else
            write(io, replace(template, "\$LABEL" => s.label))
        end
    end
    return String(take!(io))
end

"""
    mask_lines(lines; kwargs...) -> Vector{String}

Batched form of [`mask_line`]. Uses [`LogClustering.Rust.parse_lines`]
to compile the regex battery once and stream every line through in
one FFI crossing — 10–50× faster than the per-line loop for typical
2 k-line benchmarks.
"""
function mask_lines(lines::AbstractVector{<:AbstractString};
                    labels::AbstractVector{<:AbstractString} = DEFAULT_LABELS,
                    patterns::AbstractVector{<:AbstractString} = DEFAULT_PATTERNS,
                    template::AbstractString = "<\$LABEL>")
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))
    all_spans = Rust.parse_lines(lines, labels, patterns)
    out = Vector{String}(undef, length(lines))
    @inbounds for (i, (line, spans)) in enumerate(zip(lines, all_spans))
        out[i] = _render_mask(String(line), spans, template)
    end
    return out
end

function _render_mask(line::AbstractString,
                      spans::AbstractVector{<:Any},
                      template::AbstractString)
    io = IOBuffer()
    @inbounds for s in spans
        if s.label === nothing
            write(io, SubString(line, s.start, s.stop))
        else
            write(io, replace(template, "\$LABEL" => s.label))
        end
    end
    return String(take!(io))
end

"""
    mask_line_with_values(line; labels = DEFAULT_LABELS,
                          patterns = DEFAULT_PATTERNS,
                          template = "<\$LABEL>")
        -> (template::String, values::Vector{SlotValue})

Same masking as [`mask_line`], plus every matched span is returned as
a [`SlotValue`]. Downstream the template goes to the clustering /
parser side (stable across value variation) and the values go to the
embedding / anomaly side (so a never-before-seen IP or an out-of-range
number still contributes a signal).

```julia
tpl, vals = mask_line_with_values("user 42 from 10.0.0.1")
# tpl  = "user <INT> from <IP>"
# vals = [SlotValue("INT", "42", 6, 7), SlotValue("IP", "10.0.0.1", 14, 21)]
```
"""
function mask_line_with_values(line::AbstractString;
                               labels::AbstractVector{<:AbstractString} = DEFAULT_LABELS,
                               patterns::AbstractVector{<:AbstractString} = DEFAULT_PATTERNS,
                               template::AbstractString = "<\$LABEL>")
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))
    spans = Rust.parse_line(line, labels, patterns)
    io = IOBuffer()
    values = SlotValue[]
    @inbounds for s in spans
        if s.label === nothing
            write(io, SubString(line, s.start, s.stop))
        else
            write(io, replace(template, "\$LABEL" => s.label))
            push!(values, SlotValue(
                String(s.label),
                String(SubString(line, s.start, s.stop)),
                s.start, s.stop,
            ))
        end
    end
    return String(take!(io)), values
end

"""
    mask_lines_with_values(lines; kwargs...)
        -> (templates::Vector{String}, values::Vector{Vector{SlotValue}})

Batched [`mask_line_with_values`]. Parallel vectors: `templates[i]`
pairs with `values[i]`, one slot per match in the original line.
Uses the batched FFI path for 10–50× the per-line throughput.
"""
function mask_lines_with_values(lines::AbstractVector{<:AbstractString};
                                labels::AbstractVector{<:AbstractString} = DEFAULT_LABELS,
                                patterns::AbstractVector{<:AbstractString} = DEFAULT_PATTERNS,
                                template::AbstractString = "<\$LABEL>")
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))
    all_spans = Rust.parse_lines(lines, labels, patterns)
    templates = Vector{String}(undef, length(lines))
    values    = Vector{Vector{SlotValue}}(undef, length(lines))
    @inbounds for (i, (line, spans)) in enumerate(zip(lines, all_spans))
        templates[i], values[i] = _render_mask_with_values(String(line), spans, template)
    end
    return templates, values
end

function _render_mask_with_values(line::AbstractString,
                                  spans::AbstractVector{<:Any},
                                  template::AbstractString)
    io = IOBuffer()
    vs = SlotValue[]
    @inbounds for s in spans
        if s.label === nothing
            write(io, SubString(line, s.start, s.stop))
        else
            write(io, replace(template, "\$LABEL" => s.label))
            push!(vs, SlotValue(
                String(s.label),
                String(SubString(line, s.start, s.stop)),
                s.start, s.stop,
            ))
        end
    end
    return String(take!(io)), vs
end

end # module Masking
