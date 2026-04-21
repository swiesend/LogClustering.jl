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

    @testset "train without --out errors" begin
        r = _with_captured_stdio(() -> CLI.main(["train",
            "--kind", "drain", "--data", "nonexistent"]))
        @test r.code == 2
        @test occursin("--out", r.err)
    end

    @testset "deep_kate train + classify round-trip over raw lines" begin
        data = _toy_file()
        model = tempname() * ".jld2"
        out   = tempname() * ".tsv"
        try
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "deep_kate", "--data", data, "--out", model,
                "--epochs", "3", "--batch", "4", "--auto", "--quiet"]))
            @test r1.code == 0
            @test isfile(model)

            r2 = _with_captured_stdio(() -> CLI.main(["classify",
                "--model", model, "--data", data, "--out", out]))
            @test r2.code == 0
            rows = readlines(out)
            @test length(rows) == length(TOY_LINES) + 1     # header + data
            # Cluster labels look like "cluster-<id>".
            @test all(row -> occursin("cluster-", split(row, '\t')[3]),
                      rows[2:end])
        finally
            for p in (data, model, out); isfile(p) && rm(p); end
        end
    end

    @testset "vq_vae train + classify round-trip" begin
        data = _toy_file()
        model = tempname() * ".jld2"
        out   = tempname() * ".tsv"
        try
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "vq_vae", "--data", data, "--out", model,
                "--epochs", "5", "--batch", "4", "--auto", "--quiet"]))
            @test r1.code == 0

            r2 = _with_captured_stdio(() -> CLI.main(["classify",
                "--model", model, "--data", data, "--out", out]))
            @test r2.code == 0
            rows = readlines(out)
            @test length(rows) == length(TOY_LINES) + 1
            @test all(row -> occursin("code-", split(row, '\t')[3]),
                      rows[2:end])
        finally
            for p in (data, model, out); isfile(p) && rm(p); end
        end
    end

    @testset "seq_lstm train + classify round-trip" begin
        data = _toy_file()
        model = tempname() * ".jld2"
        out   = tempname() * ".tsv"
        try
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "seq_lstm", "--data", data, "--out", model,
                "--epochs", "3", "--batch", "4", "--seqlen", "6",
                "--auto", "--quiet"]))
            @test r1.code == 0

            r2 = _with_captured_stdio(() -> CLI.main(["classify",
                "--model", model, "--data", data, "--out", out]))
            @test r2.code == 0
            rows = readlines(out)
            @test length(rows) == length(TOY_LINES) + 1
        finally
            for p in (data, model, out); isfile(p) && rm(p); end
        end
    end

    # -------------------------------------------------------------
    # score — load a saved ValueNoveltyDetector and emit per-line
    # anomaly scores. Round-trips the detector through
    # PersistenceGlue.save / load_and_rehydrate.
    # -------------------------------------------------------------

    @testset "score — TSV round-trip through a saved ValueNoveltyDetector" begin
        using LogClustering.Instance: ValueNoveltyDetector, update!
        using LogClustering.Masking: mask_lines_with_values
        using LogClustering.PersistenceGlue: PersistenceGlue

        # Train the detector on a 3-line fixture so it remembers an IP
        # it has seen; score a 4-line fixture where one line sports a
        # brand-new IP. The novelty score for that line must be strictly
        # greater than for the three familiar ones.
        train_lines = ["user 1 from 10.0.0.1",
                       "user 2 from 10.0.0.1",
                       "user 3 from 10.0.0.1"]
        _, train_vals = mask_lines_with_values(train_lines)
        det = ValueNoveltyDetector()
        for vs in train_vals; update!(det, vs); end

        det_path  = tempname() * ".jld2"
        data_path = tempname() * ".log"
        out_path  = tempname() * ".tsv"
        try
            PersistenceGlue.save(det_path, det)

            open(data_path, "w") do io
                println(io, "user 4 from 10.0.0.1")   # known IP
                println(io, "user 5 from 10.0.0.1")   # known IP
                println(io, "user 6 from 10.0.0.1")   # known IP
                println(io, "user 7 from 99.99.99.99") # novel IP
            end

            r = _with_captured_stdio(() -> CLI.main(["score",
                "--detector", det_path,
                "--data", data_path,
                "--out", out_path,
                "--format", "tsv"]))
            @test r.code == 0

            lines = readlines(out_path)
            @test lines[1] == "line_id\tscore\tline"
            @test length(lines) == 5                          # header + 4 rows
            scores = [parse(Float64, split(ln, '\t')[2]) for ln in lines[2:end]]
            @test all(isfinite, scores)
            @test scores[4] > scores[1]                       # novel > known
        finally
            for p in (det_path, data_path, out_path); isfile(p) && rm(p); end
        end
    end

    @testset "score --format json emits parseable NDJSON" begin
        using LogClustering.Instance: ValueNoveltyDetector, update!
        using LogClustering.Masking: mask_lines_with_values
        using LogClustering.PersistenceGlue: PersistenceGlue

        lines_train = ["user 1 from 10.0.0.1", "user 2 from 10.0.0.1"]
        _, vs = mask_lines_with_values(lines_train)
        det = ValueNoveltyDetector()
        for v in vs; update!(det, v); end

        det_path  = tempname() * ".jld2"
        data_path = tempname() * ".log"
        out_path  = tempname() * ".jsonl"
        try
            PersistenceGlue.save(det_path, det)
            open(data_path, "w") do io
                println(io, "user 3 from 10.0.0.1")
                println(io, "user 4 from 8.8.8.8")
            end

            r = _with_captured_stdio(() -> CLI.main(["score",
                "--detector", det_path,
                "--data", data_path,
                "--out", out_path,
                "--format", "json"]))
            @test r.code == 0

            records = [JSON3.read(l) for l in readlines(out_path)]
            @test length(records) == 2
            @test haskey(records[1], :line_id)
            @test haskey(records[1], :score)
            @test haskey(records[1], :line)
        finally
            for p in (det_path, data_path, out_path); isfile(p) && rm(p); end
        end
    end

    @testset "score without --detector exits 2 with ArgumentError" begin
        r = _with_captured_stdio(() -> CLI.main(["score"]))
        @test r.code == 2
        @test occursin("--detector", r.err)
    end

    # -------------------------------------------------------------
    # benchmark — wraps benchmarks/loghub2/run.jl. We exercise the
    # --help path (no FS side effects) and a single-dataset end-to-end
    # run against the tiny checked-in toy_structured.csv fixture. No
    # network traffic either way.
    # -------------------------------------------------------------

    @testset "benchmark --help prints usage from run.jl" begin
        r = _with_captured_stdio(() -> CLI.main(["benchmark", "--help"]))
        @test r.code == 0
        @test occursin("usage:", r.out)
        @test occursin("parsers", r.out) || occursin("parser", r.out)
    end

    @testset "benchmark — end-to-end on the checked-in toy fixture" begin
        toy = joinpath(@__DIR__, "..", "benchmarks", "loghub2",
                       "toy_structured.csv")
        if !isfile(toy)
            @test_skip isfile(toy)
        else
            r = _with_captured_stdio(() -> CLI.main(["benchmark", toy, "identity"]))
            @test r.code == 0
            # format_report emits one row; our test just checks we got
            # non-empty output and no thrown error.
            @test !isempty(strip(r.out))
        end
    end

    # -------------------------------------------------------------
    # download-loghub — wraps benchmarks/loghub2/download.jl.
    # `--help` is the only guaranteed side-effect-free invocation;
    # `unknown-dataset` asserts the bad-input error path returns
    # non-zero *without* ever touching the network.
    # -------------------------------------------------------------

    @testset "download-loghub --help lists datasets + flags" begin
        r = _with_captured_stdio(() -> CLI.main(["download-loghub", "--help"]))
        @test r.code == 0
        @test occursin("--all", r.out)
        @test occursin("HDFS", r.out)
        @test occursin("Apache", r.out)
    end

    @testset "download-loghub rejects unknown dataset name" begin
        r = _with_captured_stdio(() -> CLI.main(["download-loghub",
                                                 "NonexistentDatasetXYZ"]))
        @test r.code != 0
    end

    # -------------------------------------------------------------
    # select-model / train --reuse — corpus fingerprint registry.
    # -------------------------------------------------------------

    @testset "select-model --help exits 0" begin
        r = _with_captured_stdio(() -> CLI.main(["select-model", "--help"]))
        @test r.code == 0
        @test occursin("registry", r.out)
    end

    @testset "select-model finds a fingerprint match in the registry" begin
        registry = mktempdir()
        data = _toy_file()
        out  = tempname() * ".jld2"
        try
            # Train once into the registry so a bundle is cached.
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "deep_kate", "--data", data,
                "--out", joinpath(registry, "seed.jld2"),
                "--epochs", "2", "--batch", "8", "--lr", "0.01",
                "--min-count", "1", "--max-vocab", "64",
                "--quiet"]))
            @test r1.code == 0

            # select-model returns that exact path for the same corpus.
            r2 = _with_captured_stdio(() -> CLI.main(["select-model",
                "--kind", "deep_kate", "--data", data,
                "--registry", registry, "--min-count", "1"]))
            @test r2.code == 0
            @test occursin("seed.jld2", r2.out)

            # A different corpus (change one line) → fingerprint mismatch.
            data2 = tempname() * ".log"
            open(data2, "w") do io
                for l in TOY_LINES; println(io, l); end
                println(io, "NEW totally different line with unseen tokens")
            end
            try
                r3 = _with_captured_stdio(() -> CLI.main(["select-model",
                    "--kind", "deep_kate", "--data", data2,
                    "--registry", registry, "--min-count", "1"]))
                @test r3.code == 1
                @test occursin("no deep_kate bundle matches", r3.err)
            finally
                isfile(data2) && rm(data2)
            end
        finally
            isfile(data) && rm(data)
            isfile(out) && rm(out)
            rm(registry; recursive = true, force = true)
        end
    end

    @testset "train --reuse short-circuits when a match exists" begin
        registry = mktempdir()
        data = _toy_file()
        out1 = joinpath(registry, "seed.jld2")
        out2 = tempname() * ".jld2"
        try
            # Seed the registry with a trained bundle.
            r1 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "deep_kate", "--data", data, "--out", out1,
                "--epochs", "2", "--batch", "8", "--lr", "0.01",
                "--min-count", "1", "--max-vocab", "64",
                "--quiet"]))
            @test r1.code == 0
            @test isfile(out1)

            # --reuse on the same corpus must find the seed and copy it.
            r2 = _with_captured_stdio(() -> CLI.main(["train",
                "--kind", "deep_kate", "--data", data, "--out", out2,
                "--epochs", "2", "--batch", "8", "--lr", "0.01",
                "--min-count", "1", "--max-vocab", "64",
                "--reuse", "--registry", registry]))
            @test r2.code == 0
            @test isfile(out2)
            @test occursin("reuse:", r2.err)
            @test filesize(out1) == filesize(out2)
        finally
            isfile(data) && rm(data)
            isfile(out2) && rm(out2)
            rm(registry; recursive = true, force = true)
        end
    end
end
