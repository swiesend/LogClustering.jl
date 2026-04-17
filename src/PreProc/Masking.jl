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

export DEFAULT_LABELS, DEFAULT_PATTERNS, mask_line, mask_lines

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
    "TIMESTAMP",   # ISO-8601 / RFC-3339
    "IPV6",
    "IP",
    "MAC",
    "UUID",
    "SHA",         # 40- or 64-hex-char checksum
    "HEX",         # 0x-prefixed hex address
    "URL",
    "EMAIL",
    "PATH",        # unix path
    "QUOTED",      # "…" double-quoted string (no embedded escapes handled)
    "DURATION",    # 42ms / 13.5s / 1h / 2m
    "SIZE",        # 12KB / 4.5GiB
    "NUM",
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
    raw"\b\d+(?:\.\d+)?(?:ms|us|ns|μs|s|m|h|d)\b",
    raw"\b\d+(?:\.\d+)?(?:[KMGTP]i?B|[kmgtp]b)\b",
    raw"\b-?\d+(?:\.\d+)?\b",
]

@assert length(DEFAULT_LABELS) == length(DEFAULT_PATTERNS)

# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

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

Batched form of [`mask_line`]. Same keyword arguments; dispatches
row-by-row to the Rust cascade.
"""
mask_lines(lines::AbstractVector{<:AbstractString}; kwargs...) =
    [mask_line(l; kwargs...) for l in lines]

end # module Masking
