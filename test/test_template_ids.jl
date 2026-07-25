using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Persistence
using LogClustering.Drain3
using LogClustering.Drain3: Drain, process!, assign, id_sequence_matrix
using Dates: DateTime

const _TID_CORPUS = String[]
let
    for i in 1:240
        r = i % 4
        push!(_TID_CORPUS,
            r == 0 ? "user alice logged in from 10.0.0.$(i % 250)" :
            r == 1 ? "request GET /api/v1/items took $(i)ms" :
            r == 2 ? "cache miss for key session:$(i)" :
                     "worker $(i % 8) processed job $(i)")
    end
end

function _tid_file()
    path = tempname() * ".log"
    open(path, "w") do io
        for l in _TID_CORPUS
            println(io, l)
        end
    end
    return path
end

@testset "M1 template-id sequences" begin

    @testset "id_sequence_matrix — shape + DeepLog windowing" begin
        d = Drain(; depth = 4, sim_th = 0.4)
        lines = ["a b c", "d e f", "a b c", "g h i", "a b c"]
        S = id_sequence_matrix(d, lines; seqlen = 4, learn = true)
        @test size(S) == (4, length(lines))          # (seqlen, n_lines)
        # ids are cluster-id + 1 (so a real id never collides with pad_id=1);
        # the padding rows are pad_id = 1.
        @test all(x -> x >= 1, S)
        # Column j's last row is line j's own (shifted) template id, and the
        # window carries the preceding lines' ids (left-padded). Recurring
        # "a b c" maps to one id across all its occurrences.
        id_abc = S[end, 1]
        @test S[end, 3] == id_abc
        @test S[end, 5] == id_abc
        # Column 5 = ids of lines 2..5 (window of 4): [d e f, a b c, g h i, a b c]
        @test S[:, 5] == [S[end, 2], id_abc, S[end, 4], id_abc]
        # Column 1 is left-padded (only one real id, the rest pad_id=1).
        @test S[1:3, 1] == [1, 1, 1]
    end

    @testset "assign — read-only, novel → 0, matches process! ids" begin
        d = Drain(; depth = 4, sim_th = 0.4)
        for l in ["INFO started pid 1", "INFO started pid 2", "INFO started pid 3"]
            process!(d, l)
        end
        n_before = length(d.clusters)
        # A known structure resolves to an existing id without growing.
        cid = assign(d, "INFO started pid 9")
        @test cid > 0
        @test length(d.clusters) == n_before          # no growth
        # A structurally novel line has no match → 0 (padded downstream).
        @test assign(d, "totally different shape here now") == 0
        @test assign(d, "") == 0
        @test length(d.clusters) == n_before
    end

    @testset "train/classify/score round-trip + embedded Drain" begin
        data = _tid_file()
        dir = mktempdir()
        model = joinpath(dir, "tx_dec_tid.jld2")
        rc = CLI.cmd_train(String[
            "--kind", "transformer_decoder", "--tokens", "template-ids",
            "--data", data, "--out", model,
            "--seqlen", "8", "--d-model", "32", "--n-layers", "2",
            "--n-heads", "4", "--epochs", "2", "--batch", "32", "--quiet"])
        @test rc == 0

        art = Persistence.rehydrate(Persistence.load(model))
        @test art.tokens === :template_ids
        @test art.drain !== nothing
        @test art.vocab === nothing                    # self-contained, no word vocab

        # classify: labels are `template-<id>`.
        cls = joinpath(dir, "cls.tsv")
        @test CLI.cmd_classify(String["--model", model, "--data", data,
                                      "--out", cls, "--format", "tsv"]) == 0
        rows = readlines(cls)
        @test length(rows) == length(_TID_CORPUS) + 1  # header + lines
        @test occursin("template-", rows[2])

        # score: finite, varying per-line NLL over whole-corpus windows.
        sco = joinpath(dir, "sco.tsv")
        @test CLI.cmd_score(String["--model", model, "--data", data,
                                   "--out", sco, "--format", "tsv"]) == 0
        srows = readlines(sco)
        @test length(srows) == length(_TID_CORPUS) + 1
        scores = [parse(Float64, split(l, '\t')[2]) for l in srows[2:end]]
        @test all(isfinite, scores)

        # Persistence: a freshly reloaded Drain assigns identical ids.
        art2 = Persistence.rehydrate(Persistence.load(model))
        probe = _TID_CORPUS[1:16]
        @test [assign(art.drain, l) for l in probe] ==
              [assign(art2.drain, l) for l in probe]
    end

    @testset "stream template-id decoder — inline history NLL" begin
        data = _tid_file()
        dir = mktempdir()
        model = joinpath(dir, "m.jld2")
        @test CLI.cmd_train(String[
            "--kind", "transformer_decoder", "--tokens", "template-ids",
            "--data", data, "--out", model,
            "--seqlen", "8", "--d-model", "32", "--n-layers", "2",
            "--n-heads", "4", "--epochs", "2", "--batch", "32", "--quiet"]) == 0

        opts = Dict{String, Any}("model" => model, "detector" => "",
                                 "dedup" => "masked", "dedup-window" => 4096,
                                 "rrcf" => false)
        inf = CLI._build_inferer(opts, :raw)
        @test CLI._is_template_id_decoder(inf)
        @test inf.nll_memo === nothing                 # memo disabled (history-dependent)

        ts = DateTime(2026, 1, 1)
        for (i, l) in enumerate(_TID_CORPUS[1:40])
            ev = LogClustering.Stream.LineEvent(l, i, ts, false)
            ir, needs_nll, _ = CLI._infer_cheap(inf, ev)
            @test needs_nll == false                   # filled inline, not deferred to batch
            @test isfinite(ir["transformer_decoder"]["nll"])
        end
        # Rolling history is bounded to seqlen.
        @test length(inf.seq_hist) == Int(inf.artifact.seqlen)
    end

    @testset "seq_lstm honours --tokens template-ids" begin
        data = _tid_file()
        dir = mktempdir()
        model = joinpath(dir, "seq_tid.jld2")
        @test CLI.cmd_train(String[
            "--kind", "seq_lstm", "--tokens", "template-ids",
            "--data", data, "--out", model,
            "--seqlen", "8", "--epochs", "2", "--batch", "32", "--quiet"]) == 0
        art = Persistence.rehydrate(Persistence.load(model))
        @test art.tokens === :template_ids
        @test art.drain !== nothing
        cls = joinpath(dir, "cls.tsv")
        @test CLI.cmd_classify(String["--model", model, "--data", data,
                                      "--out", cls, "--format", "tsv"]) == 0
        @test length(readlines(cls)) == length(_TID_CORPUS) + 1
    end

end
