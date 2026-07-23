using Test
using LogClustering
using LogClustering.Memory.RedisClient
using LogClustering.Memory.RedisClient: Client, connect_redis, call,
                                         sadd!, sismember, smembers,
                                         zadd!, zcard, zremrangebyscore!,
                                         set!, get_str, expire!, del!,
                                         ping, flushdb!, _parse_url
using LogClustering.Memory.WarmStore
using LogClustering.Memory.WarmStore: InProcWarmStore, RedisWarmStore,
                                       open_warm, healthy,
                                       seen_member, add_member!, load_set,
                                       load_sketch, merge_sketch!
using LogClustering.Rules
using LogClustering.Rules: load_rules, attach_warm_store!, evaluate,
                            persist_sketches!
using LogClustering.TDigests: TDigests
using LogClustering.Stream: LineEvent
using LogClustering.StructuredLog
using Sockets
using JSON3
using Dates: UTC, now

const _REDIS_LOG = IOBuffer()
StructuredLog.set_format!(:json; stream = _REDIS_LOG)

# ---------------------------------------------------------------------------
# In-process Redis-shaped TCP server: speaks just enough RESP-2 to
# exercise the SET / GET / SADD / SISMEMBER / SMEMBERS / DEL / PING /
# AUTH / SELECT path used by our client. Backed by a Dict + a Set
# per key prefix. Runs in a Task and tears down on `close(server)`.
# ---------------------------------------------------------------------------

mutable struct _FakeRedis
    server::Sockets.TCPServer
    port::Int
    sets::Dict{String, Set{String}}
    strings::Dict{String, String}
    task::Union{Nothing, Task}
end

function _fake_redis(; auth::String = "")
    server = Sockets.listen(Sockets.localhost, 0)
    port = Sockets.getsockname(server)[2]
    return _fake_redis_from(server, Int(port), auth)
end

# Bind a fresh fake server on a specific port (for reconnect tests).
function _fake_redis_on_port(port::Int; auth::String = "")
    server = Sockets.listen(Sockets.localhost, port)
    return _fake_redis_from(server, port, auth)
end

function _fake_redis_from(server, port::Int, auth::String)
    fr = _FakeRedis(server, port, Dict{String, Set{String}}(),
                    Dict{String, String}(), nothing)
    fr.task = @async begin
        try
            while isopen(fr.server)
                sock = accept(fr.server)
                @async _handle_client(sock, fr, auth)
            end
        catch
        end
    end
    return fr
end

Base.close(fr::_FakeRedis) = (try; close(fr.server); catch; end; nothing)

# RESP-2 inbound parser — minimal, just enough to read *N $L … from
# our own client.
function _read_resp_request(sock)
    line = _readline_crlf(sock)
    isempty(line) && return String[]
    @assert line[1] == '*'
    n = parse(Int, line[2:end])
    args = String[]
    for _ in 1:n
        l = _readline_crlf(sock)
        @assert l[1] == '$'
        len = parse(Int, l[2:end])
        bytes = read(sock, len)
        # consume CRLF
        read(sock, 2)
        push!(args, String(bytes))
    end
    return args
end

function _readline_crlf(sock)
    buf = IOBuffer()
    while !eof(sock)
        b = read(sock, UInt8)
        if b == UInt8('\r')
            read(sock, 1)
            break
        end
        write(buf, b)
    end
    return String(take!(buf))
end

function _write_ok(sock)    ; write(sock, "+OK\r\n"); end
function _write_int(sock, n); write(sock, ":", string(n), "\r\n"); end
function _write_nil(sock)    ; write(sock, "\$-1\r\n"); end
function _write_str(sock, s) ; write(sock, "\$", string(sizeof(s)), "\r\n", s, "\r\n"); end
function _write_arr(sock, a)
    write(sock, "*", string(length(a)), "\r\n")
    for s in a; _write_str(sock, String(s)); end
end
function _write_err(sock, msg); write(sock, "-", msg, "\r\n"); end

function _handle_client(sock, fr::_FakeRedis, auth::String)
    authed = isempty(auth)
    try
        while !eof(sock)
            args = _read_resp_request(sock)
            isempty(args) && break
            cmd = uppercase(args[1])
            if cmd == "AUTH"
                if length(args) >= 2 && args[2] == auth
                    authed = true; _write_ok(sock)
                else
                    _write_err(sock, "WRONGPASS")
                end
            elseif !authed
                _write_err(sock, "NOAUTH")
            elseif cmd == "PING"
                write(sock, "+PONG\r\n")
            elseif cmd == "SELECT"
                _write_ok(sock)
            elseif cmd == "SET"
                fr.strings[args[2]] = args[3]
                _write_ok(sock)
            elseif cmd == "GET"
                v = get(fr.strings, args[2], nothing)
                v === nothing ? _write_nil(sock) : _write_str(sock, v)
            elseif cmd == "DEL"
                n = 0
                for k in args[2:end]
                    if haskey(fr.strings, k); delete!(fr.strings, k); n += 1; end
                    if haskey(fr.sets, k);    delete!(fr.sets, k);    n += 1; end
                end
                _write_int(sock, n)
            elseif cmd == "SADD"
                key = args[2]
                s = get!(fr.sets, key, Set{String}())
                added = 0
                for m in args[3:end]
                    if !(m in s); push!(s, m); added += 1; end
                end
                _write_int(sock, added)
            elseif cmd == "SISMEMBER"
                s = get(fr.sets, args[2], Set{String}())
                _write_int(sock, args[3] in s ? 1 : 0)
            elseif cmd == "SMEMBERS"
                s = collect(get(fr.sets, args[2], Set{String}()))
                _write_arr(sock, s)
            elseif cmd == "EXPIRE"
                _write_int(sock, 1)
            elseif cmd == "FLUSHDB"
                empty!(fr.sets); empty!(fr.strings); _write_ok(sock)
            else
                _write_err(sock, "unknown command $cmd")
            end
        end
    catch
    finally
        try; close(sock); catch; end
    end
end

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

@testset "Memory.RedisClient — RESP-2 over fake server" begin
    fr = _fake_redis()
    try
        c = connect_redis("redis://localhost:$(fr.port)/0")

        @testset "PING + SET/GET round-trip" begin
            @test ping(c)
            @test set!(c, "k", "v") == "OK"
            @test get_str(c, "k") == "v"
            @test get_str(c, "missing") === nothing
        end

        @testset "SADD / SISMEMBER / SMEMBERS" begin
            @test sadd!(c, "myset", ["a", "b"]) == 2
            @test sadd!(c, "myset", ["a"])      == 0
            @test sismember(c, "myset", "a")
            @test !sismember(c, "myset", "z")
            @test Set(smembers(c, "myset")) == Set(["a", "b"])
        end

        @testset "DEL + FLUSHDB clears keys" begin
            set!(c, "to-del", "x")
            @test del!(c, "to-del") == 1
            sadd!(c, "purge:set", ["m"])
            @test flushdb!(c)
            @test isempty(smembers(c, "purge:set"))
        end

        @testset "reconnects after the server drops the connection" begin
            # Restart the fake server on the SAME port: the client's
            # socket is now dead but may still report open, so the next
            # command must transparently reconnect + retry once.
            port = fr.port
            close(fr)
            sleep(0.1)
            fr2 = _fake_redis_on_port(port)
            try
                sleep(0.1)
                @test set!(c, "after-restart", "v") == "OK"
                @test get_str(c, "after-restart") == "v"
            finally
                close(fr2)
            end
        end

        close(c)
    finally
        close(fr)
    end
end

@testset "Memory.RedisClient — URL parsing edge cases" begin
    @test _parse_url("redis://localhost")        == ("localhost", 6379, 0, "")
    @test _parse_url("redis://localhost/")       == ("localhost", 6379, 0, "")
    @test _parse_url("redis://localhost:6380/2") == ("localhost", 6380, 2, "")
    @test _parse_url("redis://:secret@h:6379/0") == ("h", 6379, 0, "secret")
    # IPv6 literals — brackets stripped, port split on ']'.
    @test _parse_url("redis://[::1]:6379/1")     == ("::1", 6379, 1, "")
    @test _parse_url("redis://[::1]")            == ("::1", 6379, 0, "")
    @test_throws ArgumentError _parse_url("memcached://x")
end

@testset "Memory.WarmStore — InProc + Redis adapters" begin
    @testset "InProcWarmStore is a no-op" begin
        s = InProcWarmStore()
        @test healthy(s)
        @test !seen_member(s, "r", "1")
        @test !add_member!(s, "r", "1")
        @test load_set(s, "r") == String[]
    end

    @testset "RedisWarmStore round-trip via fake server" begin
        fr = _fake_redis()
        try
            store = open_warm("redis://localhost:$(fr.port)/0";
                              rules_fingerprint = "abcdef012345")
            @test store isa RedisWarmStore
            @test healthy(store)
            # Add three distinct ids; the third repeat is suppressed.
            @test add_member!(store, "novel_template", "10")
            @test add_member!(store, "novel_template", "11")
            @test !add_member!(store, "novel_template", "10")
            @test seen_member(store, "novel_template", "10")
            @test !seen_member(store, "novel_template", "99")
            @test Set(load_set(store, "novel_template")) == Set(["10", "11"])
            close(store)
        finally
            close(fr)
        end
    end

    @testset "open_warm rejects unsupported URL schemes" begin
        @test_throws ArgumentError open_warm("memcached://x")
    end

    @testset "sketch merge — cross-shard baseline via GET-merge-SET" begin
        fr = _fake_redis()
        try
            store = open_warm("redis://localhost:$(fr.port)/0";
                              rules_fingerprint = "sketch0000000")
            @test load_sketch(store, "score_p99") === nothing
            # Shard A contributes low half, shard B the high half.
            a = TDigests.TDigest(); b = TDigests.TDigest()
            for i in 1:5000;      push!(a, Float64(i)); end
            for i in 5001:10_000; push!(b, Float64(i)); end
            merge_sketch!(store, "score_p99", TDigests.serialize(a))
            merge_sketch!(store, "score_p99", TDigests.serialize(b))
            # The combined sketch spans both halves.
            combined = TDigests.deserialize(load_sketch(store, "score_p99"))
            @test TDigests.count(combined) == 10_000
            @test abs(TDigests.quantile(combined, 0.99) - 9900) < 300
            close(store)
        finally
            close(fr)
        end
    end
end

@testset "Rules — auto:p99 baseline hydrates from + persists to warm store" begin
    body = """
    { "version": 1,
      "defaults": { "warmup_required": false, "cooldown_s": 0 },
      "rules": [
        { "id": "p99", "kind": "score_threshold",
          "metric": "score", "comparison": ">", "value": "auto:p99" } ] }
    """
    fr = _fake_redis()
    try
        # Session 1 seeds a baseline and persists it to Redis.
        rs1 = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        store1 = open_warm("redis://localhost:$(fr.port)/0";
                           rules_fingerprint = "hydra00000000")
        attach_warm_store!(rs1, store1)
        for i in 1:2000
            evaluate(rs1, Dict{String,Any}("line"=>"x","line_id"=>i,
                                            "score"=>Float64(i)))
        end
        persist_sketches!(rs1)
        @test load_sketch(store1, "p99") !== nothing

        # Session 2 (fresh rule set) hydrates that baseline at attach —
        # so a large value fires immediately without re-warming.
        rs2 = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        store2 = open_warm("redis://localhost:$(fr.port)/0";
                           rules_fingerprint = "hydra00000000")
        attach_warm_store!(rs2, store2)
        # p99 of 1..2000 ≈ 1980; a spike well above must fire on the first line.
        t = evaluate(rs2, Dict{String,Any}("line"=>"x","line_id"=>1,
                                            "score"=>50_000.0))
        @test length(t) == 1
        close(store1); close(store2)
    finally
        close(fr)
    end
end

@testset "Rules.attach_warm_store! — novel_cluster persistence" begin
    body = """
    { "version": 1,
      "defaults": { "warmup_required": false, "cooldown_s": 0 },
      "rules": [
        { "id": "nc", "kind": "novel_cluster", "model": "drain" } ] }
    """
    rs = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)

    fr = _fake_redis()
    try
        store = open_warm("redis://localhost:$(fr.port)/0";
                          rules_fingerprint = "deadbeef0000")
        attach_warm_store!(rs, store)

        # First run — cluster 7 is novel.
        ir = Dict{String, Any}("line" => "x", "line_id" => 1,
                                "drain" => Dict("cluster_id" => 7))
        t = evaluate(rs, ir)
        @test length(t) == 1
        @test seen_member(store, "nc", "7")

        # Simulate a restart: build a fresh rule set, attach the same
        # store; the second run should NOT fire again.
        rs2 = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        attach_warm_store!(rs2, store)
        ir2 = Dict{String, Any}("line" => "x", "line_id" => 1,
                                 "drain" => Dict("cluster_id" => 7))
        @test isempty(evaluate(rs2, ir2))
        # A genuinely new cluster id still fires.
        ir3 = Dict{String, Any}("line" => "x", "line_id" => 2,
                                 "drain" => Dict("cluster_id" => 8))
        @test length(evaluate(rs2, ir3)) == 1

        close(store)
    finally
        close(fr)
    end
end
