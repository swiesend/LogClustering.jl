using Test
using LogClustering
using LogClustering.Rules: SinkSpec
using LogClustering.SinksWebhook
using LogClustering.SinksWebhook: WebhookSink, WebhookStats,
                                    deliver!, enqueue!, format_body,
                                    substitute_env, start_webhook_task,
                                    _redacted_url
using LogClustering.StructuredLog
using JSON3

# Pipe logger to a buffer so the @async retry warnings don't blow up on
# Test's redirected stderr.
const _LOG_SINK = IOBuffer()
StructuredLog.set_format!(:json; stream = _LOG_SINK)

# Build a SinkSpec without going through the JSON parser.
function _sink(; url::String = "https://example.com/hook",
                format::Symbol = :raw,
                headers::Dict{String, String} = Dict{String, String}(),
                secret_env::String = "")
    spec = SinkSpec("webhook:test", :webhook, url, "POST", headers,
                    secret_env, format)
    return WebhookSink(spec = spec, max_attempts = 3,
                       base_backoff_s = 0.01, max_backoff_s = 0.05)
end

# Mock HTTP request callable. Records calls onto `calls`, returns the
# response taken from `responses`. Each response can be a NamedTuple
# (status, headers) or a function `(method, url, headers, body) -> resp`.
mutable struct MockResp
    status::Int
    headers::Dict{String, String}
    body::String
end
MockResp(status::Int; body::String = "", headers = Dict{String,String}()) =
    MockResp(status, headers, body)

function _mock_request(calls::Vector, responses::Vector)
    idx = Ref(0)
    return function (method, url, headers, body; kwargs...)
        idx[] += 1
        push!(calls, (method = method, url = url,
                      headers = Dict(headers), body = body))
        r = responses[min(idx[], length(responses))]
        return r
    end
end

# HTTP-shaped response object the deliver loop knows how to query.
struct _Resp
    status::Int
    headers::Vector{Pair{String, String}}
end

_Resp(status::Int) = _Resp(status, Pair{String,String}[])

import HTTP
HTTP.header(r::_Resp, name::AbstractString, default::AbstractString = "") =
    something(findfirst(p -> lowercase(p.first) == lowercase(name), r.headers),
              nothing) === nothing ? default :
    r.headers[findfirst(p -> lowercase(p.first) == lowercase(name), r.headers)].second

@testset "SinksWebhook" begin

    @testset "format_body — :raw mirrors event" begin
        s = _sink(format = :raw)
        ev = Dict("event" => "trigger", "rule_id" => "fatal",
                  "severity" => "crit", "line_id" => 7, "line" => "boom")
        body = format_body(s, ev)
        d = JSON3.read(body)
        @test String(d["rule_id"]) == "fatal"
        @test Int(d["line_id"])     == 7
    end

    @testset "format_body — :slack truncates" begin
        s = _sink(format = :slack)
        long = repeat("x", 2000)
        ev = Dict("rule_id" => "f", "severity" => "warn",
                  "line_id" => 1, "line" => long)
        d = JSON3.read(format_body(s, ev))
        @test occursin("[warn]", String(d["text"]))
        @test sizeof(String(d["text"])) < 2200   # truncated, not raw 2k
    end

    @testset "format_body — :alertmanager schema" begin
        s = _sink(format = :alertmanager)
        ev = Dict("rule_id" => "oom", "severity" => "crit",
                  "line_id" => 9, "ts" => "2026-04-27T14:02:11.842Z",
                  "line" => "kernel: OOM")
        d = JSON3.read(format_body(s, ev))
        @test length(d) == 1
        @test String(d[1]["labels"]["alertname"]) == "oom"
        @test String(d[1]["labels"]["severity"])  == "crit"
        @test occursin("OOM", String(d[1]["annotations"]["description"]))
        @test String(d[1]["startsAt"]) == "2026-04-27T14:02:11.842Z"
    end

    @testset "substitute_env replaces \${env:VAR}" begin
        env = Dict("TOK" => "secret123", "OTHER" => "ok")
        out = substitute_env("Bearer \${env:TOK} and \${env:OTHER}"; env = env)
        @test out == "Bearer secret123 and ok"
        # Missing var -> empty + warning (we just check substitution).
        out = substitute_env("\${env:DOES_NOT_EXIST}"; env = env)
        @test out == ""
    end

    @testset "deliver! — happy path (200 OK)" begin
        calls = []
        s = _sink()
        s.request = _mock_request(calls, [_Resp(200)])
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test ok
        @test length(calls) == 1
        @test s.stats.sent_total == 1
        @test s.stats.failed_total == 0
    end

    @testset "deliver! — secret_env injected as Authorization" begin
        calls = []
        env = Dict("TOKEN" => "live_abc")
        s = _sink(secret_env = "TOKEN")
        s.request = _mock_request(calls, [_Resp(200)])
        deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                         "line_id" => 1, "line" => "hi"); env = env)
        @test length(calls) == 1
        h = calls[1].headers
        @test get(h, "Authorization", "") == "Bearer live_abc"
    end

    @testset "deliver! — 503 then 200 retries to success" begin
        calls = []
        s = _sink()
        s.request = _mock_request(calls,
            [_Resp(503), _Resp(503), _Resp(200)])
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test ok
        @test length(calls) == 3
        @test s.stats.sent_total == 1
        @test s.stats.retried_total == 2
    end

    @testset "deliver! — gives up after max_attempts" begin
        calls = []
        s = _sink()
        s.request = _mock_request(calls,
            [_Resp(503), _Resp(503), _Resp(503), _Resp(503)])
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test !ok
        @test length(calls) == 3   # max_attempts
        @test s.stats.failed_total == 1
    end

    @testset "deliver! — 429 with Retry-After respected" begin
        calls = []
        s = _sink()
        # First response: 429 with a tiny Retry-After we can observe in
        # the warn log. Second: 200.
        s.request = _mock_request(calls,
            [_Resp(429, ["Retry-After" => "0.02"]), _Resp(200)])
        t0 = time()
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        elapsed = time() - t0
        @test ok
        @test length(calls) == 2
        @test elapsed >= 0.01    # honoured the Retry-After sleep
    end

    @testset "deliver! — 400 fails immediately (no retry)" begin
        calls = []
        s = _sink()
        s.request = _mock_request(calls,
            [_Resp(400), _Resp(400), _Resp(400)])
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test !ok
        @test length(calls) == 1
        @test s.stats.failed_total == 1
    end

    @testset "deliver! — connect errors retry then give up" begin
        calls = []
        s = _sink()
        s.request = function (method, url, headers, body; kwargs...)
            push!(calls, true)
            error("connect refused")
        end
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test !ok
        @test length(calls) == s.max_attempts
        @test s.stats.failed_total == 1
    end

    # Real HTTP.jl THROWS StatusError for non-2xx unless
    # status_exception=false is passed; both stacks must classify the
    # status identically. These mocks throw like the real default.
    @testset "deliver! — thrown StatusError(401) fails without retry" begin
        calls = []
        s = _sink()
        s.request = function (method, url, headers, body; kwargs...)
            push!(calls, true)
            throw(HTTP.StatusError(401, "POST", "/", _Resp(401)))
        end
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test !ok
        @test length(calls) == 1          # permanent 4xx — no retries
        @test s.stats.failed_total == 1
        @test s.stats.retried_total == 0
    end

    @testset "deliver! — thrown StatusError(503) retries then 200 succeeds" begin
        calls = []
        s = _sink()
        n = Ref(0)
        s.request = function (method, url, headers, body; kwargs...)
            push!(calls, true)
            n[] += 1
            n[] == 1 && throw(HTTP.StatusError(503, "POST", "/", _Resp(503)))
            return _Resp(200)
        end
        ok = deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                              "line_id" => 1, "line" => "hi"))
        @test ok
        @test length(calls) == 2
        @test s.stats.retried_total == 1
    end

    @testset "deliver! — passes status_exception=false to the client" begin
        seen = Ref{Any}(nothing)
        s = _sink()
        s.request = function (method, url, headers, body; kwargs...)
            seen[] = Dict(kwargs)
            return _Resp(200)
        end
        deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                         "line_id" => 1, "line" => "hi"))
        @test seen[][:status_exception] == false
    end

    @testset "enqueue! — drops on full / closed, counts, never blocks" begin
        s = _sink()
        ch = Channel{Dict{String, Any}}(2)
        ev = Dict("rule_id" => "x")
        @test enqueue!(ch, ev, s)
        @test enqueue!(ch, ev, s)
        t0 = time()
        @test !enqueue!(ch, ev, s)         # full → immediate false
        @test time() - t0 < 0.5            # and it did not block
        @test s.stats.dropped_total == 1
        close(ch)
        @test !enqueue!(ch, ev, s)         # closed → false, no throw
        @test s.stats.dropped_total == 2
    end

    @testset "start_webhook_task drains queued events on close" begin
        delivered = []
        s = _sink()
        s.request = function (method, url, headers, body; kwargs...)
            push!(delivered, body); return _Resp(200)
        end
        ch, task, stop = start_webhook_task(s; capacity = 8)
        for i in 1:3
            enqueue!(ch, Dict("rule_id" => "r$i", "severity" => "warn",
                              "line_id" => i, "line" => "L"), s)
        end
        close(ch)                          # no stop flag — drain expected
        @test timedwait(() -> istaskdone(task), 5.0) === :ok
        @test length(delivered) == 3
    end

    @testset "_redacted_url strips path, query, and userinfo" begin
        @test _redacted_url("https://hooks.slack.com/services/T00/B00/secret") ==
              "https://hooks.slack.com/…"
        @test _redacted_url("https://user:pw@host.example:8443/x?token=abc") ==
              "https://host.example:8443/…"
        @test _redacted_url("host.example/path") == "host.example/…"
    end

    @testset "failure logs carry only the redacted URL" begin
        buf = IOBuffer()
        StructuredLog.set_format!(:json; stream = buf)
        try
            s = _sink(url = "https://hooks.slack.com/services/T00/B00/secretpart")
            s.request = (args...; kwargs...) -> _Resp(400)
            deliver!(s, Dict("rule_id" => "x", "severity" => "warn",
                             "line_id" => 1, "line" => "hi"))
            logs = String(take!(buf))
            @test !occursin("secretpart", logs)
            @test occursin("hooks.slack.com", logs)
        finally
            StructuredLog.set_format!(:json; stream = _LOG_SINK)
        end
    end

end
