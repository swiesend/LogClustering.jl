"""
    Framing

Zero-copy, recursive-descent framing parsers for common log collector
envelopes. Used as Stage E′ step 1: strip the collector prefix so
downstream stages (masking, tokenisation, embedding) see just the
structured message body.

Grammars implemented:

- `SOURCE_SYSLOG_5424`  — RFC 5424 syslog (including recursive
  STRUCTURED-DATA with nested SD-ELEMENTs and escaped PARAM-VALUEs).
- `SOURCE_K8S_CRI`      — Kubernetes CRI log format
  (`<timestamp> <stream> <tag> <message>`).
- `SOURCE_DOCKER_JSONL` — Docker json-file driver
  (`{"log":"…","stream":"…","time":"…"}`).
- `SOURCE_RAW`          — fallback; the whole line is taken as message.

All parsers return a [`Frame`] whose string fields are
`SubString{String}` views into the caller's buffer — no strings are
allocated on the hot path. `parse_frame` dispatches on the first byte.
"""
module Framing

export Source, Frame, parse_frame,
       SOURCE_SYSLOG_5424, SOURCE_K8S_CRI, SOURCE_DOCKER_JSONL, SOURCE_RAW,
       foreach_sd_element, foreach_sd_param

@enum Source::UInt8 SOURCE_SYSLOG_5424 SOURCE_K8S_CRI SOURCE_DOCKER_JSONL SOURCE_RAW

const _EMPTY = SubString("", 1, 0)

struct Frame
    source::Source
    priority::Int16
    version::Int8
    timestamp::SubString{String}
    host::SubString{String}
    app::SubString{String}
    procid::SubString{String}
    msgid::SubString{String}
    sd::SubString{String}
    stream::SubString{String}
    tag::SubString{String}
    message::SubString{String}
end

# --- byte helpers ----------------------------------------------------------

@inline _is_digit(c::UInt8) = (c - UInt8('0')) < 0x0a

@inline function _expect(s::String, i::Int, n::Int, c::UInt8)
    i <= n && @inbounds(codeunit(s, i)) == c
end

@inline function _parse_digits(s::String, i::Int, n::Int, maxlen::Int)
    start = i
    stop = min(n, i + maxlen - 1)
    v = 0
    while i <= stop
        c = @inbounds codeunit(s, i)
        _is_digit(c) || break
        v = v * 10 + Int(c - UInt8('0'))
        i += 1
    end
    return (i == start ? nothing : v), i
end

@inline function _scan_until(s::String, i::Int, n::Int, c::UInt8)
    start = i
    while i <= n && @inbounds(codeunit(s, i)) != c
        i += 1
    end
    return SubString(s, start, i - 1), i
end

# --- public entry point ----------------------------------------------------

"""
    parse_frame(line) -> Frame

Dispatch on the first byte of `line`:

- `<`  → RFC 5424 syslog
- `{`  → Docker JSON-lines
- `0`-`9` → Kubernetes CRI
- otherwise → `SOURCE_RAW`

Falls back to `SOURCE_RAW` (whole line as `.message`) when a grammar is
attempted but fails to match.
"""
function parse_frame(line::AbstractString)::Frame
    s = line isa String ? line : String(line)
    n = sizeof(s)
    n == 0 && return _raw(s)
    c = @inbounds codeunit(s, 1)
    if c == UInt8('<')
        f = _try_syslog_5424(s, n)
        f.source == SOURCE_SYSLOG_5424 && return f
    elseif c == UInt8('{')
        f = _try_docker_jsonl(s, n)
        f.source == SOURCE_DOCKER_JSONL && return f
    elseif _is_digit(c)
        f = _try_k8s_cri(s, n)
        f.source == SOURCE_K8S_CRI && return f
    end
    return _raw(s)
end

@inline function _raw(s::String)
    Frame(SOURCE_RAW, Int16(-1), Int8(-1),
          _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY,
          SubString(s, 1))
end

# --- RFC 5424 syslog -------------------------------------------------------

function _try_syslog_5424(s::String, n::Int)::Frame
    # "<" PRIVAL ">" VERSION SP TS SP HOST SP APP SP PROCID SP MSGID SP SD [SP MSG]
    _expect(s, 1, n, UInt8('<')) || return _raw(s)
    pri, i = _parse_digits(s, 2, n, 3)
    pri === nothing && return _raw(s)
    pri <= 191 || return _raw(s)
    _expect(s, i, n, UInt8('>')) || return _raw(s)
    i += 1

    ver, i = _parse_digits(s, i, n, 3)
    ver === nothing && return _raw(s)
    1 <= ver <= 99 || return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    ts, i = _scan_until(s, i, n, UInt8(' '))
    isempty(ts) && return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    host, i = _scan_until(s, i, n, UInt8(' '))
    isempty(host) && return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    app, i = _scan_until(s, i, n, UInt8(' '))
    isempty(app) && return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    procid, i = _scan_until(s, i, n, UInt8(' '))
    isempty(procid) && return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    msgid, i = _scan_until(s, i, n, UInt8(' '))
    isempty(msgid) && return _raw(s)

    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    sd_res = _parse_sd(s, i, n)
    sd_res === nothing && return _raw(s)
    sd, i = sd_res

    msg = _EMPTY
    if i <= n
        _expect(s, i, n, UInt8(' ')) || return _raw(s)
        i += 1
        # Optional UTF-8 BOM per RFC 5424 §6.4
        if i + 2 <= n &&
           @inbounds(codeunit(s, i))   == 0xEF &&
           @inbounds(codeunit(s, i+1)) == 0xBB &&
           @inbounds(codeunit(s, i+2)) == 0xBF
            i += 3
        end
        msg = SubString(s, i, n)
    end

    return Frame(SOURCE_SYSLOG_5424, Int16(pri), Int8(ver),
                 ts, host, app, procid, msgid, sd, _EMPTY, _EMPTY, msg)
end

# STRUCTURED-DATA = NILVALUE / 1*SD-ELEMENT
function _parse_sd(s::String, i::Int, n::Int)
    if _expect(s, i, n, UInt8('-'))
        return SubString(s, i, i), i + 1
    end
    sd_start = i
    (i <= n && @inbounds(codeunit(s, i)) == UInt8('[')) || return nothing
    while i <= n && @inbounds(codeunit(s, i)) == UInt8('[')
        j = _skip_sd_element(s, i, n)
        j === nothing && return nothing
        i = j
    end
    return SubString(s, sd_start, i - 1), i
end

# SD-ELEMENT = "[" SD-ID *(SP SD-PARAM) "]"
function _skip_sd_element(s::String, i::Int, n::Int)::Union{Int, Nothing}
    @inbounds(codeunit(s, i)) == UInt8('[') || return nothing
    i += 1
    id_start = i
    while i <= n
        c = @inbounds codeunit(s, i)
        (c == UInt8(' ') || c == UInt8(']') || c == UInt8('=') || c == UInt8('"')) && break
        i += 1
    end
    i > id_start || return nothing
    while i <= n && @inbounds(codeunit(s, i)) == UInt8(' ')
        i += 1
        j = _skip_sd_param(s, i, n)
        j === nothing && return nothing
        i = j
    end
    _expect(s, i, n, UInt8(']')) || return nothing
    return i + 1
end

# SD-PARAM = PARAM-NAME "=" %x22 PARAM-VALUE %x22
function _skip_sd_param(s::String, i::Int, n::Int)::Union{Int, Nothing}
    name_start = i
    while i <= n
        c = @inbounds codeunit(s, i)
        c == UInt8('=') && break
        (c == UInt8(' ') || c == UInt8(']') || c == UInt8('"')) && return nothing
        i += 1
    end
    i > name_start || return nothing
    _expect(s, i, n, UInt8('=')) || return nothing
    i += 1
    _expect(s, i, n, UInt8('"')) || return nothing
    i += 1
    while i <= n
        c = @inbounds codeunit(s, i)
        if c == UInt8('\\')
            i + 1 <= n || return nothing
            i += 2
        elseif c == UInt8('"')
            return i + 1
        else
            i += 1
        end
    end
    return nothing
end

# --- Structured-data iteration (callback, zero-alloc) ----------------------

"""
    foreach_sd_element(f, sd)

Invoke `f(id::SubString, params::SubString)` for each SD-ELEMENT in `sd`
(the raw `.sd` field of a `SOURCE_SYSLOG_5424` [`Frame`]). `params` is the
substring between the SD-ID and the closing `]`, ready to be passed to
[`foreach_sd_param`]. No-op when `sd` is `-` (NILVALUE) or empty.
"""
function foreach_sd_element(f, sd::AbstractString)
    s = sd isa SubString{String} ? sd.string : (sd isa String ? sd : String(sd))
    off = sd isa SubString{String} ? sd.offset : 0
    len = sizeof(sd)
    n = off + len
    i = off + 1
    (len == 1 && @inbounds(codeunit(s, i)) == UInt8('-')) && return nothing
    while i <= n
        @inbounds(codeunit(s, i)) == UInt8('[') || return nothing
        i += 1
        id_start = i
        while i <= n
            c = @inbounds codeunit(s, i)
            (c == UInt8(' ') || c == UInt8(']')) && break
            i += 1
        end
        id = SubString(s, id_start, i - 1)
        params_start = i
        while i <= n && @inbounds(codeunit(s, i)) != UInt8(']')
            if @inbounds(codeunit(s, i)) == UInt8('"')
                i += 1
                while i <= n
                    c = @inbounds codeunit(s, i)
                    if c == UInt8('\\') && i + 1 <= n
                        i += 2
                    elseif c == UInt8('"')
                        i += 1
                        break
                    else
                        i += 1
                    end
                end
            else
                i += 1
            end
        end
        params = SubString(s, params_start, i - 1)
        f(id, params)
        i <= n && @inbounds(codeunit(s, i)) == UInt8(']') || return nothing
        i += 1
    end
    return nothing
end

"""
    foreach_sd_param(f, params)

Invoke `f(name::SubString, value::SubString)` for each `NAME="VALUE"` pair
in `params` (as yielded by [`foreach_sd_element`]). `value` is the raw
substring *between* the quotes; callers that need to unescape `\\"`, `\\\\`
or `\\]` should do so on demand.
"""
function foreach_sd_param(f, params::AbstractString)
    s = params isa SubString{String} ? params.string : (params isa String ? params : String(params))
    off = params isa SubString{String} ? params.offset : 0
    n = off + sizeof(params)
    i = off + 1
    while i <= n
        while i <= n && @inbounds(codeunit(s, i)) == UInt8(' ')
            i += 1
        end
        i > n && return nothing
        name_start = i
        while i <= n && @inbounds(codeunit(s, i)) != UInt8('=')
            i += 1
        end
        i <= n || return nothing
        name = SubString(s, name_start, i - 1)
        i += 1                              # skip '='
        @inbounds(codeunit(s, i)) == UInt8('"') || return nothing
        i += 1
        val_start = i
        while i <= n
            c = @inbounds codeunit(s, i)
            if c == UInt8('\\') && i + 1 <= n
                i += 2
            elseif c == UInt8('"')
                break
            else
                i += 1
            end
        end
        i <= n || return nothing
        f(name, SubString(s, val_start, i - 1))
        i += 1                              # skip closing '"'
    end
    return nothing
end

# --- Kubernetes CRI --------------------------------------------------------

function _try_k8s_cri(s::String, n::Int)::Frame
    # RFC3339Nano SP (stdout|stderr) SP (F|P) SP MSG
    ts, i = _scan_until(s, 1, n, UInt8(' '))
    (sizeof(ts) >= 20) || return _raw(s)          # at least "YYYY-MM-DDTHH:MM:SSZ"
    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    stream, i = _scan_until(s, i, n, UInt8(' '))
    (stream == "stdout" || stream == "stderr") || return _raw(s)
    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    tag, i = _scan_until(s, i, n, UInt8(' '))
    (tag == "F" || tag == "P") || return _raw(s)
    _expect(s, i, n, UInt8(' ')) || return _raw(s)
    i += 1
    msg = SubString(s, i, n)
    return Frame(SOURCE_K8S_CRI, Int16(-1), Int8(-1),
                 ts, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, stream, tag, msg)
end

# --- Docker JSON-lines -----------------------------------------------------

# Expected shape: {"log":"...","stream":"...","time":"..."}
# Keys may appear in any order. Unknown keys are skipped. String values
# only (log/stream/time are all strings in the docker json-file driver).
function _try_docker_jsonl(s::String, n::Int)::Frame
    _expect(s, 1, n, UInt8('{')) || return _raw(s)
    i = 2
    log = _EMPTY
    stream = _EMPTY
    time = _EMPTY
    while i <= n
        i = _skip_ws(s, i, n)
        i > n && return _raw(s)
        @inbounds(codeunit(s, i)) == UInt8('}') && break
        @inbounds(codeunit(s, i)) == UInt8('"') || return _raw(s)
        key_start = i + 1
        j = _skip_json_string(s, i, n)
        j === nothing && return _raw(s)
        i = j::Int
        key = SubString(s, key_start, i - 2)
        i = _skip_ws(s, i, n)
        _expect(s, i, n, UInt8(':')) || return _raw(s)
        i += 1
        i = _skip_ws(s, i, n)
        i > n && return _raw(s)
        @inbounds(codeunit(s, i)) == UInt8('"') || return _raw(s)
        val_start = i + 1
        j = _skip_json_string(s, i, n)
        j === nothing && return _raw(s)
        i = j::Int
        val = SubString(s, val_start, i - 2)
        if key == "log"
            log = val
        elseif key == "stream"
            stream = val
        elseif key == "time"
            time = val
        end
        i = _skip_ws(s, i, n)
        if i <= n && @inbounds(codeunit(s, i)) == UInt8(',')
            i += 1
        elseif i <= n && @inbounds(codeunit(s, i)) == UInt8('}')
            break
        else
            return _raw(s)
        end
    end
    isempty(time) && isempty(log) && isempty(stream) && return _raw(s)
    return Frame(SOURCE_DOCKER_JSONL, Int16(-1), Int8(-1),
                 time, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, stream, _EMPTY, log)
end

@inline function _skip_ws(s::String, i::Int, n::Int)
    while i <= n
        c = @inbounds codeunit(s, i)
        (c == UInt8(' ') || c == UInt8('\t')) || break
        i += 1
    end
    return i
end

# Given i pointing at the opening '"', consume the string and return the
# index one past the closing '"'. Handles JSON backslash escapes.
function _skip_json_string(s::String, i::Int, n::Int)::Union{Int, Nothing}
    @inbounds(codeunit(s, i)) == UInt8('"') || return nothing
    i += 1
    while i <= n
        c = @inbounds codeunit(s, i)
        if c == UInt8('\\')
            i + 1 <= n || return nothing
            i += 2
        elseif c == UInt8('"')
            return i + 1
        else
            i += 1
        end
    end
    return nothing
end

end # module Framing
