using Test
using LogClustering
using LogClustering.Rules: SinkSpec
using LogClustering.SinksWebhook
using LogClustering.SinksWebhook: WebhookSink, WebhookStats,
                                    deliver!, format_body, substitute_env
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

end
