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

export parse_line, infer_regex, Span, ABI_VERSION

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
    infer_regex(samples; replacements = String[], wildcard = ".*?",
                classes = Dict{String,String}()) -> String

Log-key regex inference (thesis Algorithm 3.10 + Stage E″ class map).
Given `samples`, a list of tokenised log-lines, return a regex that
matches every sample. Tokens that agree across every sample become
literals; tokens that differ at the same position become alternations
`(a|b|c)`.

If `replacements` is non-empty, any token containing one of those
substrings is rewritten to `wildcard` — this collapses descriptive
placeholders like `%RCE_DATETIME%` into `.*?`.

If `classes` is non-empty, it maps label names (without the `%` wrappers)
to tight regexes. A token that is exactly `%LABEL%` and has `LABEL` in
the map is replaced by the class regex rather than the default wildcard.
This is the MDL-ladder step described in plan 001 Stage E″: generic
`.*?` slots are replaced with narrower classes like
`IP → \\d{1,3}(?:\\.\\d{1,3}){3}` or `NUM → \\d+`.

Adjacent identical wildcard fragments (e.g. `.*?.*?.*?` produced by
several consecutive label tokens) are coalesced to a single copy, since
`.*?.*?` and `.*?` match the same language.

Metacharacters are escaped via the Rust `regex` crate's `escape`
routine, so the result is safe to compile with RE2/Hyperscan.
"""
function infer_regex(
    samples::AbstractVector{<:AbstractVector{<:AbstractString}};
    replacements::AbstractVector{<:AbstractString} = String[],
    wildcard::AbstractString = ".*?",
    classes::AbstractDict = Dict{String, String}(),
)::String
    samples_buf = _encode_samples(samples)
    replacements_buf = _encode_token_list(replacements)
    wildcard_bytes = codeunits(String(wildcard))
    classes_buf = _encode_classes(classes)

    lib = _lib()
    handle = Libdl.dlopen(lib)
    try
        sym_infer = Libdl.dlsym(handle, :lc_rs_infer_regex)
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
