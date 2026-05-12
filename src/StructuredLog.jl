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

const _LEVELS = (:debug => 10, :info => 20, :warn => 30, :error => 40)

const _STATE = Ref{NamedTuple{(:level, :format, :stream), Tuple{Int, Symbol, IO}}}(
    (level = 20, format = :json, stream = Base.stderr))
const _LOCK = ReentrantLock()

_level_int(s::Symbol) = something(findfirst(p -> p[1] === s, _LEVELS),
                                   nothing) === nothing ?
    error("unknown log level: $s") :
    _LEVELS[findfirst(p -> p[1] === s, _LEVELS)][2]

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
    set_format!(fmt::Symbol; stream::IO = Base.stderr)

Switch the wire format. `:json` (default) emits one JSON object per
line; `:text` falls back to a human-readable `LEVEL ts msg key=value`
form for development use.
"""
function set_format!(fmt::Symbol; stream::IO = Base.stderr)
    fmt in (:json, :text) || error("unknown log format: $fmt")
    lock(_LOCK) do
        _STATE[] = (level = _STATE[].level, format = fmt, stream = stream)
    end
    return fmt
end

_iso_now() = string(Dates.format(now(UTC), Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

_redact_key(k::AbstractString) =
    occursin(r"(?i)(authorization|token|secret|api[_-]?key|password|cookie)", k)

"Replace sensitive header / kwarg values with `***`. Operates on the
dict in place (a fresh dict is built by the caller)."
function _redact!(d::AbstractDict)
    for (k, v) in d
        ks = String(string(k))
        if _redact_key(ks)
            d[k] = "***"
        elseif v isa AbstractDict
            _redact!(v)
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
        lock(_LOCK) do
            println(state.stream, body)
            flush(state.stream)
        end
    else
        kv = IOBuffer()
        for (k, v) in kwargs
            ks = String(string(k))
            vs = _redact_key(ks) ? "***" : string(v)
            print(kv, " ", ks, "=", vs)
        end
        lock(_LOCK) do
            println(state.stream, uppercase(String(level)), " ", _iso_now(),
                    " ", msg, String(take!(kv)))
            flush(state.stream)
        end
    end
    return nothing
end

@inline info(msg::AbstractString;        kwargs...) = log_event(:info,  msg; kwargs...)
@inline warn(msg::AbstractString;        kwargs...) = log_event(:warn,  msg; kwargs...)
@inline error_event(msg::AbstractString; kwargs...) = log_event(:error, msg; kwargs...)
@inline debug(msg::AbstractString;       kwargs...) = log_event(:debug, msg; kwargs...)

end # module StructuredLog
