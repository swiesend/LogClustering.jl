"""
    LogClustering.Rust

Julia-side bindings for the companion `logclustering_rs` crate. The crate
implements two performance-critical kernels:

- [`parse_line`] — thesis Algorithm 3.1, recursive regex-cascade descent
  over a single line; returns an ordered, non-overlapping decomposition
  of the line into labelled and raw spans.
- [`infer_regex`] — thesis Algorithm 3.10, log-key regex inference over a
  set of tokenised samples.

The crate is built by `deps/build.jl`. If `cargo` is unavailable at build
time the library will not exist, and calling any function below raises
`ErrorException("logclustering_rs is not built …")`. Rebuild with
`using Pkg; Pkg.build("LogClustering")` once Rust is installed.
"""
module Rust

using Libdl

export parse_line, parse_lines, infer_regex, Span, ABI_VERSION

const EXPECTED_ABI_VERSION = UInt32(2)

# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------

const _LIB_NAMES = ("liblogclustering_rs", "logclustering_rs")

function _candidate_paths()
    root = abspath(joinpath(@__DIR__, "..", "rust", "target", "release"))
    paths = String[]
    for stem in _LIB_NAMES
        for ext in (".so", ".dylib", ".dll")
            push!(paths, joinpath(root, stem * ext))
        end
    end
    return paths
end

function _find_library()
    for p in _candidate_paths()
        isfile(p) && return p
    end
    return ""
end

const LIB_REF = Ref{String}("")

function _lib()
    isempty(LIB_REF[]) && (LIB_REF[] = _find_library())
    path = LIB_REF[]
    isempty(path) && error(
        "logclustering_rs is not built. Install Rust (https://rustup.rs) and " *
        "run `using Pkg; Pkg.build(\"LogClustering\")`.",
    )
    return path
end

function __init__()
    LIB_REF[] = _find_library()
end

"Return the Rust ABI version the crate advertises, or `0` if unbuilt."
function abi_version()::UInt32
    lib = _find_library()
    isempty(lib) && return UInt32(0)
    handle = Libdl.dlopen(lib)
    try
        sym = Libdl.dlsym(handle, :lc_rs_abi_version)
        return @ccall $sym()::UInt32
    finally
        Libdl.dlclose(handle)
    end
end

const ABI_VERSION = EXPECTED_ABI_VERSION

# ---------------------------------------------------------------------------
# parse_line
# ---------------------------------------------------------------------------

struct Span
    start::Int           # 1-based, inclusive
    stop::Int            # 1-based, inclusive
    label::Union{Nothing, String}
end

struct _CSpan
    start::UInt32        # 0-based
    stop::UInt32         # exclusive
    label_idx::Int32     # -1 for raw
end

struct _CSpans
    ptr::Ptr{_CSpan}
    len::Csize_t
    cap::Csize_t
end

struct _CLineSpan
    line_idx::UInt32
    start::UInt32
    stop::UInt32        # exclusive
    label_idx::Int32    # -1 for raw
end

struct _CLineSpans
    spans_ptr::Ptr{_CLineSpan}
    spans_len::Csize_t
    spans_cap::Csize_t
    offsets_ptr::Ptr{UInt32}
    offsets_len::Csize_t
    offsets_cap::Csize_t
end

"""
    parse_line(line::AbstractString, labels::Vector{String}, patterns::Vector{String})
        -> Vector{Span}

Recursive regex-cascade parser (thesis Algorithm 3.1) implemented in Rust.
Returns an ordered, non-overlapping decomposition of `line`. Each [`Span`]
is either a matched label (`span.label` = one of `labels`) or a raw
substring between matches (`span.label === nothing`).

`labels` and `patterns` must have equal length; `patterns[i]` is the
regex for `labels[i]`. Rank order matters — list complex patterns before
simple ones so typed slots win over generic ones.

Patterns use the Rust `regex` crate's syntax (RE2-flavoured, linear time,
no backrefs or lookaround).
"""
function parse_line(
    line::AbstractString,
    labels::AbstractVector{<:AbstractString},
    patterns::AbstractVector{<:AbstractString},
)::Vector{Span}
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))

    line_bytes = codeunits(String(line))
    n = length(labels)

    label_bytes = [codeunits(String(l)) for l in labels]
    pattern_bytes = [codeunits(String(p)) for p in patterns]

    label_ptrs = [pointer(b) for b in label_bytes]
    label_lens = [Csize_t(length(b)) for b in label_bytes]
    pattern_ptrs = [pointer(b) for b in pattern_bytes]
    pattern_lens = [Csize_t(length(b)) for b in pattern_bytes]

    lib = _lib()
    handle = Libdl.dlopen(lib)
    try
        sym_parse = Libdl.dlsym(handle, :lc_rs_parse_line)
        sym_free = Libdl.dlsym(handle, :lc_rs_free_spans)

        raw = @ccall $sym_parse(
            pointer(line_bytes)::Ptr{UInt8},
            Csize_t(length(line_bytes))::Csize_t,
            pointer(label_ptrs)::Ptr{Ptr{UInt8}},
            pointer(label_lens)::Ptr{Csize_t},
            pointer(pattern_ptrs)::Ptr{Ptr{UInt8}},
            pointer(pattern_lens)::Ptr{Csize_t},
            Csize_t(n)::Csize_t,
        )::Ptr{_CSpans}
        raw == C_NULL && error("lc_rs_parse_line failed (invalid input or regex)")
        try
            spans = unsafe_load(raw)
            out = Vector{Span}(undef, spans.len)
            for i in 1:spans.len
                cs = unsafe_load(spans.ptr, i)
                label = cs.label_idx < 0 ? nothing : String(labels[cs.label_idx + 1])
                # Convert 0-based half-open to 1-based inclusive byte indices.
                out[i] = Span(Int(cs.start) + 1, Int(cs.stop), label)
            end
            return out
        finally
            @ccall $sym_free(raw::Ptr{_CSpans})::Cvoid
        end
    finally
        Libdl.dlclose(handle)
    end
end

# ---------------------------------------------------------------------------
# infer_regex
# ---------------------------------------------------------------------------

struct _CBytes
    ptr::Ptr{UInt8}
    len::Csize_t
    cap::Csize_t
end

"""
    parse_lines(lines, labels, patterns) -> Vector{Vector{Span}}

Batched form of [`parse_line`]. Compiles each regex once and runs
the cascade over every line in `lines`, returning one span vector
per input line. Cuts the per-line FFI crossing cost that dominates
`PreProc.Masking.mask_lines` on large corpora — on a 2 k line
benchmark the call goes from ~11 s to well under 1 s.

Semantics are identical to calling [`parse_line`] in a loop: the
`i`-th output is `parse_line(lines[i], labels, patterns)`.
"""
function parse_lines(
    lines::AbstractVector{<:AbstractString},
    labels::AbstractVector{<:AbstractString},
    patterns::AbstractVector{<:AbstractString},
)::Vector{Vector{Span}}
    length(labels) == length(patterns) ||
        throw(ArgumentError("labels and patterns must have equal length"))

    # Pack lines into a single (u32 num_lines) + (u32 len, bytes)* buffer.
    # `codeunits` is UTF-8 view; `GC.@preserve` keeps the original
    # Strings alive for the FFI call's lifetime.
    line_bytes_list = [codeunits(String(l)) for l in lines]
    lines_buf = _encode_strings(line_bytes_list)

    n = length(labels)
    label_bytes   = [codeunits(String(l)) for l in labels]
    pattern_bytes = [codeunits(String(p)) for p in patterns]
    label_ptrs    = [pointer(b) for b in label_bytes]
    label_lens    = [Csize_t(length(b)) for b in label_bytes]
    pattern_ptrs  = [pointer(b) for b in pattern_bytes]
    pattern_lens  = [Csize_t(length(b)) for b in pattern_bytes]

    lib = _lib()
    handle = Libdl.dlopen(lib)
    try
        sym_parse = Libdl.dlsym(handle, :lc_rs_parse_lines)
        sym_free = Libdl.dlsym(handle, :lc_rs_free_line_spans)

        raw = @ccall $sym_parse(
            pointer(lines_buf)::Ptr{UInt8},
            Csize_t(length(lines_buf))::Csize_t,
            pointer(label_ptrs)::Ptr{Ptr{UInt8}},
            pointer(label_lens)::Ptr{Csize_t},
            pointer(pattern_ptrs)::Ptr{Ptr{UInt8}},
            pointer(pattern_lens)::Ptr{Csize_t},
            Csize_t(n)::Csize_t,
        )::Ptr{_CLineSpans}
        raw == C_NULL && error("lc_rs_parse_lines failed (invalid input or regex)")
        try
            info = unsafe_load(raw)
            # offsets has length == num_lines + 1, forming a prefix sum
            # into the flat span buffer.
            num_lines = Int(info.offsets_len) - 1
            num_lines == length(lines) || error(
                "lc_rs_parse_lines returned $(num_lines + 1) offsets for " *
                "$(length(lines)) input lines",
            )
            offsets = Vector{Int}(undef, info.offsets_len)
            @inbounds for i in 1:info.offsets_len
                offsets[i] = Int(unsafe_load(info.offsets_ptr, i))
            end
            out = Vector{Vector{Span}}(undef, num_lines)
            @inbounds for i in 1:num_lines
                from = offsets[i] + 1     # 1-based inclusive
                to   = offsets[i + 1]     # 1-based inclusive
                k    = to - from + 1
                line_spans = Vector{Span}(undef, max(k, 0))
                for j in 1:k
                    cs = unsafe_load(info.spans_ptr, from + j - 1)
                    label = cs.label_idx < 0 ? nothing :
                            String(labels[cs.label_idx + 1])
                    # 0-based half-open → 1-based inclusive.
                    line_spans[j] = Span(Int(cs.start) + 1, Int(cs.stop), label)
                end
                out[i] = line_spans
            end
            return out
        finally
            @ccall $sym_free(raw::Ptr{_CLineSpans})::Cvoid
        end
    finally
        Libdl.dlclose(handle)
    end
end

# Wire format: u32 num_strings + (u32 len + bytes)*.
function _encode_strings(xs::AbstractVector{<:AbstractVector{UInt8}})
    total = 4 + sum(4 + length(b) for b in xs; init = 0)
    out = Vector{UInt8}(undef, total)
    cur = 1
    _write_u32!(out, cur, UInt32(length(xs))); cur += 4
    @inbounds for b in xs
        _write_u32!(out, cur, UInt32(length(b))); cur += 4
        copyto!(out, cur, b, 1, length(b)); cur += length(b)
    end
    return out
end

@inline function _write_u32!(buf::Vector{UInt8}, at::Int, v::UInt32)
    @inbounds begin
        buf[at]     =  v        & 0xff
        buf[at + 1] = (v >> 8)  & 0xff
        buf[at + 2] = (v >> 16) & 0xff
        buf[at + 3] = (v >> 24) & 0xff
    end
end

"""
    infer_regex(samples; replacements = String[], wildcard = ".*?",
                classes = Dict{String,String}(), align = false) -> String

Log-key regex inference (thesis Algorithm 3.10 + Stage E″ class map
+ optional anti-unification).

Given `samples`, a list of tokenised log-lines, return a regex that
matches every sample. By default — `align = false` — samples must be
position-aligned (every sample has the same length); tokens that agree
across every sample become literals, tokens that differ at the same
position become alternations `(a|b|c)`.

Pass `align = true` to run anti-unification first: variable-length
samples are folded pairwise via LCS, runs of unmatched tokens collapse
to a single wildcard. This is Stage E″ "anti-unification over aligned
tokens" from plan 001.

If `replacements` is non-empty, any token containing one of those
substrings is rewritten to `wildcard` — this collapses descriptive
placeholders like `%RCE_DATETIME%` into `.*?`.

If `classes` is non-empty, it maps label names (without the `%` wrappers)
to tight regexes. A token that is exactly `%LABEL%` and has `LABEL` in
the map is replaced by the class regex rather than the default wildcard.

Adjacent identical wildcard fragments are coalesced.

Metacharacters are escaped via the Rust `regex` crate's `escape`
routine; the result is safe to compile with RE2/Hyperscan.
"""
function infer_regex(
    samples::AbstractVector{<:AbstractVector{<:AbstractString}};
    replacements::AbstractVector{<:AbstractString} = String[],
    wildcard::AbstractString = ".*?",
    classes::AbstractDict = Dict{String, String}(),
    align::Bool = false,
)::String
    samples_buf = _encode_samples(samples)
    replacements_buf = _encode_token_list(replacements)
    wildcard_bytes = codeunits(String(wildcard))
    classes_buf = _encode_classes(classes)

    sym_name = align ? :lc_rs_infer_regex_aligned : :lc_rs_infer_regex

    lib = _lib()
    handle = Libdl.dlopen(lib)
    try
        sym_infer = Libdl.dlsym(handle, sym_name)
        sym_free = Libdl.dlsym(handle, :lc_rs_free_bytes)

        raw = @ccall $sym_infer(
            pointer(samples_buf)::Ptr{UInt8},
            Csize_t(length(samples_buf))::Csize_t,
            pointer(replacements_buf)::Ptr{UInt8},
            Csize_t(length(replacements_buf))::Csize_t,
            pointer(wildcard_bytes)::Ptr{UInt8},
            Csize_t(length(wildcard_bytes))::Csize_t,
            pointer(classes_buf)::Ptr{UInt8},
            Csize_t(length(classes_buf))::Csize_t,
        )::Ptr{_CBytes}
        raw == C_NULL && error("lc_rs_infer_regex failed (malformed input)")
        try
            b = unsafe_load(raw)
            return unsafe_string(b.ptr, b.len)
        finally
            @ccall $sym_free(raw::Ptr{_CBytes})::Cvoid
        end
    finally
        Libdl.dlclose(handle)
    end
end

function _encode_classes(classes::AbstractDict)
    out = UInt8[]
    _push_u32!(out, UInt32(length(classes)))
    for (k, v) in classes
        kb = codeunits(String(k))
        vb = codeunits(String(v))
        _push_u32!(out, UInt32(length(kb)))
        append!(out, kb)
        _push_u32!(out, UInt32(length(vb)))
        append!(out, vb)
    end
    return out
end

# Wire format: see rust/src/lib.rs — little-endian u32 counts.
function _encode_samples(samples)
    out = UInt8[]
    _push_u32!(out, UInt32(length(samples)))
    for s in samples
        _push_u32!(out, UInt32(length(s)))
        for w in s
            wb = codeunits(String(w))
            _push_u32!(out, UInt32(length(wb)))
            append!(out, wb)
        end
    end
    return out
end

function _encode_token_list(tokens)
    out = UInt8[]
    _push_u32!(out, UInt32(length(tokens)))
    for t in tokens
        tb = codeunits(String(t))
        _push_u32!(out, UInt32(length(tb)))
        append!(out, tb)
    end
    return out
end

@inline function _push_u32!(buf::Vector{UInt8}, v::UInt32)
    push!(buf, v & 0xff)
    push!(buf, (v >> 8) & 0xff)
    push!(buf, (v >> 16) & 0xff)
    push!(buf, (v >> 24) & 0xff)
    return buf
end

end # module Rust
