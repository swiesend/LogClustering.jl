using Test
using LogClustering
using LogClustering.Stream
using LogClustering.Stream: LineEvent, TailStats, StopSignal,
                             stream_stdin!, tail_file!, stop!, is_stopped
using LogClustering.StructuredLog
using JSON3

# Drain a channel synchronously, with a hard timeout, so tests can't hang.
function _drain(ch::Channel{LineEvent}; timeout_s::Float64 = 5.0)
    events = LineEvent[]
    deadline = time() + timeout_s
    while time() < deadline
        if isready(ch)
            push!(events, take!(ch))
        elseif !isopen(ch) && !isready(ch)
            break
        else
            sleep(0.01)
        end
    end
    return events
end

# Test runners (and `Pkg.test`) redirect stderr, which breaks the
# logger's cached IO when the @async tailer task tries to log a
# rotation. Park the log output in a discardable IOBuffer for the
# duration of these tests.
const _LOG_SINK = IOBuffer()
StructuredLog.set_format!(:json; stream = _LOG_SINK)

@testset "Stream — producers" begin

    @testset "stream_stdin!: lines pushed onto channel with monotonic ids" begin
        io = IOBuffer("alpha\nbeta\ngamma\n")
        ch = Channel{LineEvent}(8)
        stats = stream_stdin!(io, ch)
        close(ch)
        evs = _drain(ch)
        @test [e.line    for e in evs] == ["alpha", "beta", "gamma"]
        @test [e.line_id for e in evs] == [1, 2, 3]
        @test stats.lines_total == 3
        @test stats.dropped_oversize == 0
    end

    @testset "stream_stdin!: oversize lines dropped" begin
        big = repeat("x", 100)
        io = IOBuffer("ok\n$big\nstill_ok\n")
        ch = Channel{LineEvent}(8)
        stats = stream_stdin!(io, ch; max_line_bytes = 50)
        close(ch)
        evs = _drain(ch)
        @test [e.line for e in evs] == ["ok", "still_ok"]
        # The oversize line consumes a line_id even though it's dropped.
        @test [e.line_id for e in evs] == [1, 3]
        @test stats.dropped_oversize == 1
    end

    @testset "tail_file!: from_start emits every line then idles at EOF" begin
        mktemp() do path, io
            write(io, "one\ntwo\nthree\n")
            flush(io)
            close(io)

            ch    = Channel{LineEvent}(16)
            stop  = StopSignal()
            stats = TailStats()
            t = @async tail_file!(path, ch; from_start = true,
                                  stop = stop, stats = stats,
                                  poll_seconds = 0.02)
            sleep(0.2)
            stop!(stop)
            wait(t)
            close(ch)
            evs = _drain(ch)
            @test [e.line for e in evs] == ["one", "two", "three"]
            @test stats.lines_total == 3
        end
    end

    @testset "tail_file!: detects new lines appended after open" begin
        path, io = mktemp()
        try
            write(io, "first\n"); flush(io)
            ch    = Channel{LineEvent}(16)
            stop  = StopSignal()
            stats = TailStats()
            t = @async tail_file!(path, ch; from_start = true,
                                  stop = stop, stats = stats,
                                  poll_seconds = 0.02)
            sleep(0.1)
            write(io, "second\n"); flush(io)
            sleep(0.15)
            write(io, "third\n"); flush(io)
            sleep(0.15)
            stop!(stop)
            wait(t)
            close(ch)
            evs = _drain(ch)
            @test [e.line for e in evs] == ["first", "second", "third"]
        finally
            close(io); rm(path; force = true)
        end
    end

    @testset "tail_file!: rotation by replace (inode change)" begin
        mktempdir() do dir
            path = joinpath(dir, "log.txt")
            open(path, "w") do io
                write(io, "a1\na2\n")
            end

            ch    = Channel{LineEvent}(16)
            stop  = StopSignal()
            stats = TailStats()
            t = @async tail_file!(path, ch; from_start = true,
                                  stop = stop, stats = stats,
                                  poll_seconds = 0.02)
            sleep(0.15)

            # logrotate `create` mode: mv original aside, create a new file
            # at the same path with fresh content.
            mv(path, path * ".1"; force = true)
            open(path, "w") do io
                write(io, "b1\nb2\n")
            end
            sleep(0.25)
            stop!(stop)
            wait(t)
            close(ch)
            evs = _drain(ch)
            @test ["a1", "a2", "b1", "b2"] ⊆ [e.line for e in evs]
            @test stats.rotations >= 1
        end
    end

    @testset "tail_file!: truncation triggers re-open" begin
        mktempdir() do dir
            path = joinpath(dir, "log.txt")
            open(path, "w") do io
                write(io, "x1\nx2\nx3\n")
            end

            ch    = Channel{LineEvent}(16)
            stop  = StopSignal()
            stats = TailStats()
            t = @async tail_file!(path, ch; from_start = true,
                                  stop = stop, stats = stats,
                                  poll_seconds = 0.02)
            sleep(0.15)

            # Truncate in place, then write a fresh shorter payload.
            open(path, "w") do io
                write(io, "fresh\n")
            end
            sleep(0.25)
            stop!(stop)
            wait(t)
            close(ch)
            evs = _drain(ch)
            @test "fresh" in [e.line for e in evs]
            @test stats.rotations >= 1
        end
    end

end

@testset "StructuredLog — JSON stderr logger" begin

    @testset "info / warn emit one JSON object per call" begin
        io = IOBuffer()
        StructuredLog.set_format!(:json; stream = io)
        StructuredLog.set_level!(:debug)
        try
            StructuredLog.info("started"; models = 2)
            StructuredLog.warn("line dropped"; bytes = 1048577,
                               reason = "oversize")
            lines = filter(!isempty,
                           split(String(take!(io)), '\n'))
            @test length(lines) == 2
            j1 = JSON3.read(lines[1])
            @test String(j1["level"]) == "info"
            @test String(j1["msg"])   == "started"
            @test Int(j1["models"])   == 2
            @test haskey(j1, "ts")
            j2 = JSON3.read(lines[2])
            @test String(j2["msg"]) == "line dropped"
            @test Int(j2["bytes"])  == 1048577
        finally
            StructuredLog.set_format!(:json; stream = Base.stderr)
        end
    end

    @testset "redacts sensitive keys" begin
        io = IOBuffer()
        StructuredLog.set_format!(:json; stream = io)
        try
            StructuredLog.info("posting"; Authorization = "Bearer s3cret",
                               api_key = "live_xx", normal = "value")
            j = JSON3.read(String(take!(io)))
            @test String(j["Authorization"]) == "***"
            @test String(j["api_key"])       == "***"
            @test String(j["normal"])        == "value"
        finally
            StructuredLog.set_format!(:json; stream = Base.stderr)
        end
    end

    @testset "redaction copies nested dicts — caller's dict untouched" begin
        io = IOBuffer()
        StructuredLog.set_format!(:json; stream = io)
        try
            headers = Dict("Authorization" => "Bearer live-secret",
                           "Content-Type"  => "application/json")
            StructuredLog.info("sending"; headers = headers)
            j = JSON3.read(String(take!(io)))
            @test String(j["headers"]["Authorization"]) == "***"
            # The caller's live dict must NOT have been redacted.
            @test headers["Authorization"] == "Bearer live-secret"
        finally
            StructuredLog.set_format!(:json; stream = _LOG_SINK)
        end
    end

    @testset "unknown level raises the descriptive error" begin
        err = try
            StructuredLog.set_level!(:trace)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("unknown log level", err.msg)
    end

    @testset "logging never throws on a broken stream" begin
        broken = IOBuffer()
        close(broken)
        StructuredLog.set_format!(:json; stream = broken)
        try
            # Must silently drop, not propagate into the caller.
            @test StructuredLog.info("into the void") === nothing
        finally
            StructuredLog.set_format!(:json; stream = _LOG_SINK)
        end
    end

    @testset "level gating drops messages below threshold" begin
        io = IOBuffer()
        StructuredLog.set_format!(:json; stream = io)
        StructuredLog.set_level!(:warn)
        try
            StructuredLog.debug("noise")
            StructuredLog.info("also noise")
            StructuredLog.warn("audible")
            StructuredLog.error_event("audible too")
            lines = filter(!isempty,
                           split(String(take!(io)), '\n'))
            @test length(lines) == 2
        finally
            StructuredLog.set_level!(:info)
            StructuredLog.set_format!(:json; stream = Base.stderr)
        end
    end

end
