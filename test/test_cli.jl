using Test
using JSON3
using LogClustering
using LogClustering.CLI
using LogClustering.Persistence

const TOY_LINES = [
    "INFO started pid 1001",
    "INFO started pid 1002",
    "ERROR connection refused from 10.0.0.1",
    "ERROR connection refused from 10.0.0.2",
    "INFO service ready",
]

function _with_captured_stdio(f)
    # Capture stdout + stderr to strings; the CLI writes to both.
    orig_out, orig_err = stdout, stderr
    rr_out = Pipe(); Base.link_pipe!(rr_out, reader_supports_async = true,
                                     writer_supports_async = true)
    rr_err = Pipe(); Base.link_pipe!(rr_err, reader_supports_async = true,
                                     writer_supports_async = true)
    redirect_stdout(rr_out)
    redirect_stderr(rr_err)
    code = try
        f()
    finally
        redirect_stdout(orig_out); redirect_stderr(orig_err)
        close(rr_out.in); close(rr_err.in)
    end
    return (
        code = code,
        out  = read(rr_out, String),
        err  = read(rr_err, String),
    )
end

function _with_stdin_file(path, f)
    orig = stdin
    open(path, "r") do io
        redirect_stdin(io)
        try
            f()
        finally
            redirect_stdin(orig)
        end
    end
end

function _toy_file()
    path = tempname() * ".log"
    open(path, "w") do io
        for l in TOY_LINES
            println(io, l)
        end
    end
    return path
end

@testset "CLI" begin
    @testset "top-level --help exits 0 and lists subcommands" begin
        r = _with_captured_stdio(() -> CLI.main(["--help"]))
        @test r.code == 0
        @test occursin("commands:", r.out)
        @test occursin("train", r.out)
        @test occursin("classify", r.out)
        @test occursin("mask", r.out)
    end

    @testset "unknown command exits 2" begin
        r = _with_captured_stdio(() -> CLI.main(["not-a-command"]))
        @test r.code == 2
        @test occursin("unknown command", r.err)
    end

    @testset "flag parser" begin
        # --flag value
        opts = CLI.parse_flags(["--a", "1", "--b", "two"],
                              [("a", 0, :int), ("b", "", :string)])
        @test opts["a"] == 1 && opts["b"] == "two"
        # --flag=value
        opts = CLI.parse_flags(["--a=3"], [("a", 0, :int)])
        @test opts["a"] == 3
        # Boolean
        opts = CLI.parse_flags(["--flag"], [("flag", false, :bool)])
        @test opts["flag"] == true
        # Unknown flag / missing value → ArgumentError.
        @test_throws ArgumentError CLI.parse_flags(["--missing"], [("a", 0, :int)])
        @test_throws ArgumentError CLI.parse_flags(["--a"], [("a", 0, :int)])
    end

    @testset "mask subcommand — stdout path" begin
        path = _toy_file()
        out_path = tempname() * ".txt"
        try
            r = _with_captured_stdio(() -> CLI.main(["mask", "--data", path,
                                                    "--out", out_path]))
            @test r.code == 0
            body = read(out_path, String)
            @test occursin("<IP>", body)
            @test occursin("<INT>", body)
            @test count(c -> c == '\n', body) == length(TOY_LINES)
        finally
            isfile(path) && rm(path)
            isfile(out_path) && rm(out_path)
        end
    end

    @testset "mask --values emits one JSON object per line" begin
        path = _toy_file()
        out_path = tempname() * ".jsonl"
        try
            r = _with_captured_stdio(() -> CLI.main(["mask", "--data", path,
                                                    "--out", out_path,
                                                    "--values"]))
            @test r.code == 0
            records = [JSON3.read(l) for l in readlines(out_path)]
            @test length(records) == length(TOY_LINES)
            @test haskey(records[1], :template)
            @test haskey(records[1], :values)
        finally
            isfile(path) && rm(path)
            isfile(out_path) && rm(out_path)
        end
    end

    @testset "train + classify — Drain round-trip through JLD2" begin
        data = _toy_file()
        model = tempname() * ".jld2"
        out  = tempname() * ".tsv"
        try
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "drain", "--data", data, "--out", model]))
            @test r1.code == 0
            @test isfile(model)

            r2 = _with_captured_stdio(() -> CLI.main(["classify",
                "--model", model, "--data", data, "--out", out]))
            @test r2.code == 0
            rows = readlines(out)
            @test length(rows) == length(TOY_LINES) + 1   # header + data
            @test rows[1] == "line_id\tcluster_id\ttemplate\tline"
            # Two INFO lines share a cluster id; 10.0.0.1 / 10.0.0.2 share another.
            cid = row -> parse(Int, split(row, '\t')[2])
            @test cid(rows[2]) == cid(rows[3])
            @test cid(rows[4]) == cid(rows[5])
            @test cid(rows[2]) != cid(rows[4])
        finally
            for p in (data, model, out)
                isfile(p) && rm(p)
            end
        end
    end

    @testset "classify --format json" begin
        data = _toy_file()
        model = tempname() * ".jld2"
        out  = tempname() * ".jsonl"
        try
            _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "drain", "--data", data, "--out", model]))
            r = _with_captured_stdio(() -> CLI.main(["classify",
                "--model", model, "--data", data, "--out", out,
                "--format", "json"]))
            @test r.code == 0
            records = [JSON3.read(l) for l in readlines(out)]
            @test length(records) == length(TOY_LINES)
            @test haskey(records[1], :cluster_id)
            @test haskey(records[1], :template)
        finally
            for p in (data, model, out)
                isfile(p) && rm(p)
            end
        end
    end

    @testset "train surfaces a helpful error for API-only kinds" begin
        data = _toy_file()
        out  = tempname() * ".jld2"
        try
            r = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "deep_kate", "--data", data, "--out", out]))
            @test r.code == 3                 # "API-only" exit code
            @test occursin("featurised", r.err)
        finally
            isfile(data) && rm(data)
            isfile(out) && rm(out)
        end
    end

    @testset "train without --out errors" begin
        r = _with_captured_stdio(() -> CLI.main(["train",
            "--kind", "drain", "--data", "nonexistent"]))
        @test r.code == 2
        @test occursin("--out", r.err)
    end
end
