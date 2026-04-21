using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.Featurise: build_vocab, bow
using LogClustering.DeepKATE: deep_kate, deep_kate_loss
using LogClustering.Instance: ValueNoveltyDetector, update!
using LogClustering.Masking: mask_lines_with_values
using LogClustering.RCA: RCA, root_cause, render_markdown, RCAReport

const RNG_RCA = Random.MersenneTwister(37)

function _sgd_step_rca(ps, grads, lr)
    grads === nothing && return ps
    ps isa AbstractArray && return ps .- lr .* grads
    if ps isa NamedTuple
        k = keys(ps)
        return NamedTuple{k}(map(ki -> _sgd_step_rca(getfield(ps, ki),
                                                     hasproperty(grads, ki) ? getfield(grads, ki) : nothing,
                                                     lr), k))
    end
    return ps
end

function _build_fixture()
    # Small corpus — two normal templates + one anomalous template.
    base_ok    = ["INFO request $(i) ok"        for i in 1:15]
    base_login = ["INFO user $(i) logged in"    for i in 1:15]
    anomalies  = ["ERROR db connection lost"    for _ in 1:3]
    lines = vcat(base_ok, base_login, anomalies)
    shuffle!(RNG_RCA, lines)
    return lines
end

function _train_quick_deep_kate(lines)
    vocab = build_vocab(lines; mask = true, min_count = 1)
    X = bow(lines, vocab; normalise = :l1)
    model = deep_kate(size(X, 1); hidden = [8, 4], latent = 3, k1 = 3)
    ps, st = Lux.setup(RNG_RCA, model)
    for _ in 1:4
        perm = shuffle(RNG_RCA, collect(1:size(X, 2)))
        for start in 1:16:size(X, 2)
            stop = min(start + 15, size(X, 2))
            xb = X[:, perm[start:stop]]
            (loss, _), back = Zygote.pullback(
                p -> deep_kate_loss(model, p, st, xb), ps)
            g = back((one(loss), nothing))[1]
            ps = _sgd_step_rca(ps, g, lr_const())
        end
    end
    return (model = model, ps = ps, st = st, vocab = vocab, X = X)
end

lr_const() = 0.05f0

@testset "RCA" begin
    lines = _build_fixture()
    art = _train_quick_deep_kate(lines)

    # One report shared across test sets — episode mining is the
    # expensive step; tight knobs keep it cheap on this fixture
    # (min_sup=3, max_gap=3, max_time_duration=5).
    rep = root_cause(art.model, art.ps, art.st, art.vocab, lines;
                     top_percentile = 0.15, min_sup = 3,
                     max_gap = 3, max_time_duration = 5,
                     k_clusters = 4)

    @testset "root_cause — basic shape + invariants" begin
        @test rep isa RCAReport
        @test length(rep.cluster_ids) == length(lines)
        @test length(rep.per_line_score) == length(lines)
        @test all(>=(0), rep.per_line_score)
        @test rep.metadata["n_lines"] == length(lines)
        @test rep.metadata["n_clusters"] == 4
        @test 0 < rep.metadata["n_anomalies"] <= length(lines)
        @test rep.threshold <= maximum(rep.per_line_score)
        if length(rep.ranked) >= 2
            @test rep.ranked[1].score >= rep.ranked[2].score
        end
    end

    @testset "root_cause — with ValueNoveltyDetector fuses value signal" begin
        _, vals = mask_lines_with_values(lines)
        det = ValueNoveltyDetector()
        for i in eachindex(lines)
            occursin("ok", lines[i]) || occursin("logged in", lines[i]) || continue
            update!(det, vals[i])
        end
        rep2 = root_cause(art.model, art.ps, art.st, art.vocab, lines;
                          detector = det, top_percentile = 0.15,
                          min_sup = 3, max_gap = 3, max_time_duration = 5,
                          k_clusters = 4)
        @test rep2.metadata["with_detector"] == true
        @test rep2.metadata["n_anomalies"] >= 1
    end

    @testset "root_cause — rejects bad arguments" begin
        @test_throws ArgumentError root_cause(art.model, art.ps, art.st,
            art.vocab, String[]; top_percentile = 0.1,
            min_sup = 3, max_gap = 3, max_time_duration = 5)
        @test_throws ArgumentError root_cause(art.model, art.ps, art.st,
            art.vocab, lines; top_percentile = 0.0,
            min_sup = 3, max_gap = 3, max_time_duration = 5)
        @test_throws ArgumentError root_cause(art.model, art.ps, art.st,
            art.vocab, lines; top_percentile = 1.0,
            min_sup = 3, max_gap = 3, max_time_duration = 5)
    end

    @testset "render_markdown — emits a readable report" begin
        md = render_markdown(rep; topk = 3, lines = lines)
        @test occursin("# RCA report", md)
        @test occursin("clusters:", md)
        @test occursin("Top ", md) && occursin("root-cause episodes", md)
    end
end
