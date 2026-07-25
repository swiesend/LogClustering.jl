"""
    StructuredLog

Tiny JSON-on-stderr logger for the streaming CLI. One JSON object per
line, one key per call-site kwarg, fixed level/ts/msg keys at the
front. No allocation-heavy macros, no global mutable handler tree —
just a thread-safe `Base.println` on stderr behind a `ReentrantLock`.

Stderr is the operator's channel; stdout stays reserved for the
[`Rules`] trigger / status / shutdown JSON-line contract. Use the
text fallback (`--log-format text`) in dev.

## Surface

- [`set_level!`] / [`set_format!`] — process-wide toggles set from
  CLI flags at boot.
- [`log_event`] / [`info`] / [`warn`] / [`error_event`] / [`debug`]
  — emit one structured record. Kwargs become top-level JSON fields
  (after the fixed `level` / `ts` / `msg` triple).
"""
module StructuredLog

using JSON3
using Dates: Dates, DateTime, now, UTC

export set_level!, set_format!, log_event, info, warn, error_event, debug

const _LEVEL_MAP = Dict{Symbol, Int}(
    :debug => 10, :info => 20, :warn => 30, :error => 40)

# `stream === nothing` means "resolve `Base.stderr` at call time".
# Never store `Base.stderr` here at module scope: the handle captured
# during precompilation is stale in every fresh process, and a logger
# that throws on its default configuration poisons every caller's
# error path.
const _STATE = Ref{NamedTuple{(:level, :format, :stream),
                              Tuple{Int, Symbol, Union{IO, Nothing}}}}(
    (level = 20, format = :json, stream = nothing))
const _LOCK = ReentrantLock()

function _level_int(s::Symbol)
    v = get(_LEVEL_MAP, s, nothing)
    v === nothing && error("unknown log level: $s (use debug|info|warn|error)")
    return v
end

"""
    set_level!(level::Symbol)

Process-wide minimum log level. Anything below is dropped.
Accepts `:debug`, `:info`, `:warn`, `:error`.
"""
function set_level!(level::Symbol)
    lock(_LOCK) do
        _STATE[] = merge(_STATE[], (level = _level_int(level),))
    end
    return level
end

"""
    set_format!(fmt::Symbol; stream::Union{IO, Nothing} = nothing)

Switch the wire format. `:json` (default) emits one JSON object per
line; `:text` falls back to a human-readable `LEVEL ts msg key=value`
form for development use. `stream = nothing` (the default) writes to
the *current* `Base.stderr`, resolved on every call — pass an
explicit `IO` to capture output (tests do this).
"""
function set_format!(fmt::Symbol; stream::Union{IO, Nothing} = nothing)
    fmt in (:json, :text) || error("unknown log format: $fmt")
    lock(_LOCK) do
        _STATE[] = (level = _STATE[].level, format = fmt, stream = stream)
    end
    return fmt
end

_iso_now() = string(Dates.format(now(UTC), Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

_redact_key(k::AbstractString) =
    occursin(r"(?i)(authorization|token|secret|api[_-]?key|password|cookie)", k)

"Replace sensitive header / kwarg values with `***`. The top-level
dict is mutated in place (it is always freshly built by `log_event`),
but nested dicts are **copied** before redaction so a caller's live
dict — e.g. an HTTP headers map about to be sent — is never
destroyed by logging it."
function _redact!(d::AbstractDict)
    for (k, v) in d
        ks = String(string(k))
        if _redact_key(ks)
            d[k] = "***"
        elseif v isa AbstractDict
            d[k] = _redact!(Dict{Any, Any}(pairs(v)))
        end
    end
    return d
end

"""
    log_event(level::Symbol, msg::AbstractString; kwargs...)

Emit one structured record at `level`. Kwargs are added as top-level
JSON fields after the fixed `level` / `ts` / `msg` triple. Returns
`nothing`. Sensitive keys (matching `authorization`, `token`,
`secret`, `api_key`, `password`, `cookie`) are redacted.
"""
function log_event(level::Symbol, msg::AbstractString; kwargs...)
    state = _STATE[]
    _level_int(level) < state.level && return nothing
    io = state.stream === nothing ? Base.stderr : state.stream

    if state.format === :json
        d = Dict{String, Any}(
            "level" => String(level),
            "ts"    => _iso_now(),
            "msg"   => String(msg),
        )
        for (k, v) in kwargs
            d[String(k)] = v
        end
        _redact!(d)
        body = JSON3.write(d)
        _write_line(io, body)
    else
        kv = IOBuffer()
        for (k, v) in kwargs
            ks = String(string(k))
            vs = _redact_key(ks) ? "***" : string(v)
            print(kv, " ", ks, "=", vs)
        end
        _write_line(io, string(uppercase(String(level)), " ", _iso_now(),
                               " ", msg, String(take!(kv))))
    end
    return nothing
end

# Logging must never throw into a caller's error path — a broken or
# redirected-away stream drops the line instead of propagating.
function _write_line(io::IO, body::AbstractString)
    try
        lock(_LOCK) do
            println(io, body)
            flush(io)
        end
    catch
    end
    return nothing
end

@inline info(msg::AbstractString;        kwargs...) = log_event(:info,  msg; kwargs...)
@inline warn(msg::AbstractString;        kwargs...) = log_event(:warn,  msg; kwargs...)
@inline error_event(msg::AbstractString; kwargs...) = log_event(:error, msg; kwargs...)
@inline debug(msg::AbstractString;       kwargs...) = log_event(:debug, msg; kwargs...)

end # module StructuredLog
