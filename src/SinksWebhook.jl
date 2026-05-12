"""
    SinksWebhook

HTTP webhook delivery for the streaming pipeline. Wraps `HTTP.jl`
with our own retry loop (exponential backoff, capped, honouring
`Retry-After`), three built-in body formats (`:raw`, `:slack`,
`:alertmanager`), and `\${env:VAR}` substitution in URLs and headers
so the operator can keep secrets in `EnvironmentFile=` rather than
in the rules JSON.

`stdout` is the primary, durable sink. Webhook delivery is best-
effort; when the queue is full the producer **drops** the delivery
(with a stderr warning) so a back-pressured endpoint cannot block
the inference loop.

## Surface

- [`WebhookSink`] — runtime state for one channel-fed delivery
  worker.
- [`start_webhook_task`] — spawn the delivery task, return the
  channel + a stop signal.
- [`deliver!`] — synchronous single-shot delivery (used for tests
  and for the `cmd_rules --dry-run` path).
- [`format_body`] — shape a `TriggerEvent`-style dict into the
  sink's chosen JSON format.
- [`substitute_env`] — replace `\${env:VAR}` placeholders.
"""
module SinksWebhook

using HTTP
using JSON3
using Dates: Dates, DateTime, now, UTC
using ..Rules: Rules, SinkSpec, TriggerEvent
using ..StructuredLog: StructuredLog

export WebhookSink, WebhookStats, start_webhook_task, deliver!,
       format_body, substitute_env

# ---------------------------------------------------------------------------
# Types.
# ---------------------------------------------------------------------------

Base.@kwdef mutable struct WebhookStats
    sent_total::Int      = 0
    failed_total::Int    = 0
    retried_total::Int   = 0
    dropped_total::Int   = 0
end

"""
    WebhookSink

A delivery worker bound to one `SinkSpec`. Tracks send counters and
holds the HTTP `request` callable so tests can inject a mock.
"""
Base.@kwdef mutable struct WebhookSink
    spec::SinkSpec
    stats::WebhookStats = WebhookStats()
    request::Function   = HTTP.request   # injectable for tests
    max_attempts::Int   = 5
    base_backoff_s::Float64 = 0.5
    max_backoff_s::Float64  = 30.0
end

# ---------------------------------------------------------------------------
# Env substitution.
# ---------------------------------------------------------------------------

const _ENV_RE = r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}"

"""
    substitute_env(s::AbstractString; env = ENV) -> String

Replace `\${env:VAR}` placeholders in `s` with values pulled from
`env` (defaults to the process `ENV`). Missing variables expand to
the empty string and emit a stderr warning; this keeps a typo in a
config file from quietly leaking an unsubstituted placeholder.
"""
function substitute_env(s::AbstractString; env = ENV)
    return replace(String(s), _ENV_RE => function (m)
        var = match(_ENV_RE, m).captures[1]
        v = get(env, var, nothing)
        if v === nothing
            StructuredLog.warn("env substitution missed";
                               var = var)
            return ""
        end
        return String(v)
    end)
end

function _substitute_headers(headers::Dict{String, String}; env = ENV)
    out = Dict{String, String}()
    for (k, v) in headers
        out[k] = substitute_env(v; env = env)
    end
    return out
end

# ---------------------------------------------------------------------------
# Body formatting.
# ---------------------------------------------------------------------------

"""
    format_body(sink::WebhookSink, event_dict::AbstractDict) -> String

Render a stdout-shape trigger dict (`event = "trigger"`, `rule_id`,
`severity`, `line_id`, `line`, `fields`, …) into the sink's
configured wire format. `event_dict` is the same JSON object the
stdout sink emits, so callers can reuse the encoder for both legs.
"""
function format_body(sink::WebhookSink, event::AbstractDict)
    if sink.spec.format === :raw
        return JSON3.write(event)
    elseif sink.spec.format === :slack
        rule_id  = String(get(event, "rule_id",  "unknown"))
        severity = String(get(event, "severity", "warn"))
        line_id  = Int(get(event, "line_id", 0))
        line     = String(get(event, "line", ""))
        if sizeof(line) > 1500
            line = String(SubString(line, 1, 1500)) * "…"
        end
        text = "<$rule_id> [$severity] line $line_id: $line"
        return JSON3.write(Dict("text" => text))
    elseif sink.spec.format === :alertmanager
        rule_id  = String(get(event, "rule_id",  "unknown"))
        severity = String(get(event, "severity", "warn"))
        ts       = String(get(event, "ts", _iso_now()))
        line     = String(get(event, "line", ""))
        line_id  = Int(get(event, "line_id", 0))
        if sizeof(line) > 1500
            line = String(SubString(line, 1, 1500)) * "…"
        end
        payload = [Dict(
            "labels" => Dict(
                "alertname" => rule_id,
                "severity"  => severity,
                "source"    => "logclustering",
            ),
            "annotations" => Dict(
                "summary"     => "$rule_id triggered at line $line_id",
                "description" => line,
            ),
            "startsAt" => ts,
        )]
        return JSON3.write(payload)
    else
        return JSON3.write(event)
    end
end

# ---------------------------------------------------------------------------
# Delivery with retries.
# ---------------------------------------------------------------------------

"""
    deliver!(sink::WebhookSink, event_dict::AbstractDict; env = ENV) -> Bool

Synchronously POST a single trigger event to `sink`. Returns `true`
on a `2xx` response, `false` after exhausting `sink.max_attempts`
retries on connect errors, `5xx`, or `429`. Permanent `4xx`
responses fail immediately without retrying.
"""
function deliver!(sink::WebhookSink, event::AbstractDict; env = ENV)
    url     = substitute_env(sink.spec.url; env = env)
    method  = sink.spec.method
    headers = _substitute_headers(sink.spec.headers; env = env)
    secret_env = sink.spec.secret_env
    if !isempty(secret_env)
        v = get(env, secret_env, nothing)
        if v !== nothing
            headers["Authorization"] = "Bearer " * String(v)
        end
    end
    body = format_body(sink, event)
    headers_pairs = collect(headers)

    backoff = sink.base_backoff_s
    for attempt in 1:sink.max_attempts
        try
            resp = sink.request(method, url, headers_pairs, body;
                                retry = false, redirect = true,
                                connect_timeout = 5.0, readtimeout = 10.0)
            status = Int(resp.status)
            if 200 <= status < 300
                sink.stats.sent_total += 1
                return true
            elseif status == 429 || (500 <= status < 600)
                # Honour Retry-After if present.
                ra = _retry_after_seconds(resp)
                sleep_s = ra > 0 ? min(ra, sink.max_backoff_s) : backoff
                if attempt < sink.max_attempts
                    sink.stats.retried_total += 1
                    StructuredLog.warn("webhook retrying";
                                       url = url, status = status,
                                       attempt = attempt,
                                       sleep_s = sleep_s)
                    sleep(sleep_s)
                    backoff = min(backoff * 2, sink.max_backoff_s)
                    continue
                else
                    sink.stats.failed_total += 1
                    StructuredLog.error_event("webhook failed";
                                               url = url, status = status,
                                               attempt = attempt)
                    return false
                end
            else
                # Permanent client error — don't retry.
                sink.stats.failed_total += 1
                StructuredLog.error_event("webhook failed";
                                           url = url, status = status,
                                           attempt = attempt)
                return false
            end
        catch e
            if attempt < sink.max_attempts
                sink.stats.retried_total += 1
                StructuredLog.warn("webhook retrying";
                                   url = url,
                                   error = sprint(showerror, e),
                                   attempt = attempt,
                                   sleep_s = backoff)
                sleep(backoff)
                backoff = min(backoff * 2, sink.max_backoff_s)
                continue
            else
                sink.stats.failed_total += 1
                StructuredLog.error_event("webhook failed";
                                           url = url,
                                           error = sprint(showerror, e),
                                           attempt = attempt)
                return false
            end
        end
    end
    return false
end

# ---------------------------------------------------------------------------
# Async delivery worker.
# ---------------------------------------------------------------------------

"""
    start_webhook_task(sink::WebhookSink;
                       capacity::Int = 1024,
                       overflow::Symbol = :drop,
                       env = ENV)
        -> (channel, task, stop_signal)

Spawn an async worker that consumes trigger events off a bounded
channel and calls [`deliver!`] for each one. When `overflow = :drop`
(default), a producer that finds the channel full skips delivery and
bumps `sink.stats.dropped_total`. With `overflow = :block` the
producer is back-pressured (use only if stdout is also blocked).

Returns the channel, the task, and a `Ref{Bool}` stop signal — set
the ref to `true` and close the channel to drain + exit cleanly.
"""
function start_webhook_task(sink::WebhookSink;
                            capacity::Int = 1024,
                            overflow::Symbol = :drop,
                            env = ENV)
    overflow in (:drop, :block) ||
        error("overflow must be :drop or :block, got :$overflow")
    ch::Channel{Dict{String, Any}} = Channel{Dict{String, Any}}(capacity)
    stop = Ref(false)
    task = @async begin
        try
            for event in ch
                stop[] && break
                deliver!(sink, event; env = env)
            end
        catch e
            StructuredLog.error_event("webhook task crashed";
                                       error = sprint(showerror, e))
        end
    end
    return (ch, task, stop)
end

"""
    enqueue!(ch::Channel, event::AbstractDict, sink::WebhookSink;
             overflow::Symbol = :drop)

Try to push `event` onto `ch`. Returns `true` on success, `false`
if the channel was full and `overflow = :drop` (bumping
`sink.stats.dropped_total`). With `overflow = :block` the call
blocks until space is available.
"""
function enqueue!(ch::Channel, event::AbstractDict, sink::WebhookSink;
                  overflow::Symbol = :drop)
    if overflow === :drop
        # Non-blocking try-put.
        if !isready(ch.cond_put) && length(ch.data) < ch.sz_max
            put!(ch, Dict{String, Any}(event))
            return true
        end
        # Best-effort try via trylock-style: rely on Channel's `put!`
        # raising when closed, otherwise drop if full.
        try
            if length(ch.data) >= ch.sz_max
                sink.stats.dropped_total += 1
                StructuredLog.warn("webhook dropped — queue full";
                                   url = sink.spec.url,
                                   capacity = ch.sz_max)
                return false
            end
            put!(ch, Dict{String, Any}(event))
            return true
        catch
            sink.stats.dropped_total += 1
            return false
        end
    else
        put!(ch, Dict{String, Any}(event))
        return true
    end
end

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

function _retry_after_seconds(resp)
    h = try
        HTTP.header(resp, "Retry-After", "")
    catch
        ""
    end
    isempty(h) && return 0.0
    n = tryparse(Float64, h)
    return n === nothing ? 0.0 : max(n, 0.0)
end

_iso_now() = string(Dates.format(now(UTC),
                                  Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

end # module SinksWebhook
