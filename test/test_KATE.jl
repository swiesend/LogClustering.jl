using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.KATE
using LogClustering.KATE: KCompetetive, count_words, normalize_log,
                          transform_text_to_input, get_similar_words

const RNG = Random.MersenneTwister(1234)

@testset "KATE" begin
    @testset "construction" begin
        l = KCompetetive(100, 8)
        @test l.in_dims == 100
        @test l.out_dims == 8
        @test l.alpha == 6.26f0
        @test_throws ArgumentError KCompetetive(10, 12)      # k > in
        @test_throws ArgumentError KCompetetive(10, 3)       # k odd
    end

    @testset "parameter shapes" begin
        l = KCompetetive(32, 4, tanh)
        ps, st = Lux.setup(RNG, l)
        @test size(ps.weight) == (4, 32)
        @test size(ps.bias) == (4,)
        @test st isa NamedTuple
    end

    @testset "forward pass (vector)" begin
        l = KCompetetive(32, 4, tanh)
        ps, st = Lux.setup(RNG, l)
        x = randn(RNG, Float32, 32)
        y, _ = l(x, ps, st)
        @test length(y) == 4
        @test all(isfinite, y)
        @test eltype(y) == Float32
    end

    @testset "forward pass (batch)" begin
        l = KCompetetive(16, 4, tanh)
        ps, st = Lux.setup(RNG, l)
        X = randn(RNG, Float32, 16, 8)
        Y, _ = l(X, ps, st)
        @test size(Y) == (4, 8)
        @test all(isfinite, Y)
    end

    @testset "competition mechanics (unit)" begin
        # All positive: half_k=2, so 2 smallest positives get zeroed, 2 largest survive.
        h = Float32[1, 2, 3, 4]
        out = KATE._apply_kcomp(h, 4, 0.0f0)   # α=0 isolates the zeroing
        @test count(iszero, out) == 2
        @test sort(collect(out)) == Float32[0, 0, 3, 4]

        # Boost: losers' energy (1+2=3) is redistributed over winners, scaled by α.
        out_boosted = KATE._apply_kcomp(h, 4, 1.0f0)
        @test out_boosted[3] ≈ 3f0 + 1f0 * (1f0 + 2f0)   # 3 + α·Epos
        @test out_boosted[4] ≈ 4f0 + 1f0 * (1f0 + 2f0)

        # Mixed polarity, even split → no competition (P==N==half_k).
        h2 = Float32[-1, -2, 1, 2]
        out2 = KATE._apply_kcomp(h2, 4, 0.0f0)
        @test out2 == h2
    end

    @testset "gradient flows through winners" begin
        l = KCompetetive(16, 4, tanh)
        ps, st = Lux.setup(RNG, l)
        x = randn(RNG, Float32, 16)
        loss(p) = sum(abs2, first(l(x, p, st)))
        g = Zygote.gradient(loss, ps)[1]
        @test g.weight isa AbstractMatrix
        @test size(g.weight) == size(ps.weight)
        @test any(!iszero, g.weight)
    end

    @testset "count_words" begin
        wc = count_words(["a", "b", "a", "c", "a", "b"])
        @test wc["a"] == 3
        @test wc["b"] == 2
        @test wc["c"] == 1
    end

    @testset "normalize_log" begin
        text = ["a", "b", "a"]
        wc = count_words(text)
        out = normalize_log(text, wc)
        @test length(out) == 3
        @test all(isfinite, out)
        @test out[1] == out[3]        # same token → same value
    end

    @testset "get_similar_words" begin
        vocab = ["alpha", "beta", "gamma", "delta"]
        W = Float32[1 0; 0.99 0.01; 0 1; -1 0]
        sims = get_similar_words(W, 1, vocab; topn = 2)
        @test length(sims) == 2
        @test sims[1] == "alpha"
        @test sims[2] == "beta"       # near-parallel to alpha
    end

    @testset "transform_text_to_input (tempfile)" begin
        mktemp() do path, io
            write(io, "alpha beta gamma alpha beta alpha")
            close(io)
            input, wc = transform_text_to_input(path; limit = 10)
            @test wc["alpha"] == 3
            @test wc["beta"] == 2
            @test wc["gamma"] == 1
            @test length(input) == 6
        end
    end
end
