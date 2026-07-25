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
       enqueue!, format_body, substitute_env

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
    # Webhook URLs are often bearer secrets themselves (Slack incoming
    # webhooks, tokens in query strings) — log scheme+host only.
    log_url = _redacted_url(url)

    backoff = sink.base_backoff_s
    for attempt in 1:sink.max_attempts
        # `status_exception = false` makes real HTTP.request return the
        # response for non-2xx instead of throwing; the catch arm also
        # classifies HTTP.StatusError so an injected/legacy stack that
        # throws still takes the right retry/no-retry branch.
        status = 0
        resp = nothing
        err = nothing
        try
            resp = sink.request(method, url, headers_pairs, body;
                                retry = false, redirect = true,
                                status_exception = false,
                                connect_timeout = 5.0, readtimeout = 10.0)
            status = Int(resp.status)
        catch e
            if e isa HTTP.StatusError
                status = Int(e.status)
                resp = e.response
            else
                err = e
            end
        end

        if err !== nothing
            # Connect / TLS / timeout error — retryable.
            if attempt < sink.max_attempts
                sink.stats.retried_total += 1
                StructuredLog.warn("webhook retrying";
                                   url = log_url,
                                   error = sprint(showerror, err),
                                   attempt = attempt,
                                   sleep_s = backoff)
                sleep(backoff)
                backoff = min(backoff * 2, sink.max_backoff_s)
                continue
            else
                sink.stats.failed_total += 1
                StructuredLog.error_event("webhook failed";
                                           url = log_url,
                                           error = sprint(showerror, err),
                                           attempt = attempt)
                return false
            end
        elseif 200 <= status < 300
            sink.stats.sent_total += 1
            return true
        elseif status == 429 || (500 <= status < 600)
            # Honour Retry-After if present.
            ra = resp === nothing ? 0.0 : _retry_after_seconds(resp)
            sleep_s = ra > 0 ? min(ra, sink.max_backoff_s) : backoff
            if attempt < sink.max_attempts
                sink.stats.retried_total += 1
                StructuredLog.warn("webhook retrying";
                                   url = log_url, status = status,
                                   attempt = attempt,
                                   sleep_s = sleep_s)
                sleep(sleep_s)
                backoff = min(backoff * 2, sink.max_backoff_s)
                continue
            else
                sink.stats.failed_total += 1
                StructuredLog.error_event("webhook failed";
                                           url = log_url, status = status,
                                           attempt = attempt)
                return false
            end
        else
            # Permanent client error — don't retry.
            sink.stats.failed_total += 1
            StructuredLog.error_event("webhook failed";
                                       url = log_url, status = status,
                                       attempt = attempt)
            return false
        end
    end
    return false
end

"""
    _redacted_url(url) -> String

Strip path, query, and userinfo from a URL for logging — Slack
incoming-webhook URLs and `?token=` query params are secrets.
"""
function _redacted_url(url::AbstractString)
    m = match(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)?(?:[^/@]*@)?([^/?#]*)", String(url))
    m === nothing && return "<invalid-url>"
    scheme = m.captures[1] === nothing ? "" : m.captures[1]
    host = m.captures[2] === nothing ? "" : m.captures[2]
    return string(scheme, host, "/…")
end

# ---------------------------------------------------------------------------
# Async delivery worker.
# ---------------------------------------------------------------------------

"""
    start_webhook_task(sink::WebhookSink;
                       capacity::Int = 1024,
                       env = ENV)
        -> (channel, task, stop_signal)

Spawn an async worker that consumes trigger events off a bounded
channel and calls [`deliver!`] for each one. Producers feed it via
[`enqueue!`], which drops (and counts) instead of blocking when the
queue is full — stdout JSON is the durable sink, webhook delivery
is best-effort.

Clean shutdown: `close(channel)` — the worker delivers everything
already queued, then exits. Setting `stop_signal[] = true` aborts
without draining.
"""
function start_webhook_task(sink::WebhookSink;
                            capacity::Int = 1024,
                            env = ENV)
    ch::Channel{Dict{String, Any}} = Channel{Dict{String, Any}}(capacity)
    stop = Ref(false)
    task = @async begin
        try
            # `close(ch)` ends this loop after the queue drains — queued
            # alerts are delivered, not dropped, at clean shutdown.
            # `stop[]` is abort-only: it short-circuits between events
            # when the operator wants an immediate exit.
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
    enqueue!(ch::Channel, event::AbstractDict, sink::WebhookSink)

Non-blocking push. Returns `true` on success, `false` when the
channel is full or closed — the delivery is dropped and counted on
`sink.stats.dropped_total` so a back-pressured endpoint can never
stall the producer. Single-producer contract: the capacity check is
check-then-act and only race-free because exactly one task feeds
the channel.
"""
function enqueue!(ch::Channel, event::AbstractDict, sink::WebhookSink)
    ok = isopen(ch) && Base.n_avail(ch) < ch.sz_max &&
         (try
              put!(ch, Dict{String, Any}(event)); true
          catch
              false
          end)
    if !ok
        sink.stats.dropped_total += 1
        StructuredLog.warn("webhook delivery dropped — queue full or closed";
                           url = _redacted_url(sink.spec.url),
                           capacity = ch.sz_max)
    end
    return ok
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
