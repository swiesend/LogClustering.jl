using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.DeepKATE
using LogClustering.DeepKATE: deep_kate, latent_layer, deep_kate_loss, repel

const RNG = Random.MersenneTwister(42)

@testset "DeepKATE" begin
    @testset "architecture matches thesis Quellcode 3.7" begin
        m = deep_kate(64; latent = 3)
        @test length(m.layers) == 10
        @test latent_layer(m) == 5

        encoder_last = m.layers[latent_layer(m)]
        @test encoder_last isa Dense
        @test encoder_last.activation === sin
        @test encoder_last.in_dims == 5
        @test encoder_last.out_dims == 3                 # latent

        decoder_last = m.layers[end]
        @test decoder_last isa Dense
        @test decoder_last.activation === sigmoid
        @test decoder_last.out_dims == 64                # matches input

        # Two KCompetetive layers with the thesis's k values.
        ks = [l.k for l in m.layers if l isa LogClustering.KATE.KCompetetive]
        @test ks == [25, 3]
    end

    @testset "forward pass (batch)" begin
        m = deep_kate(32; latent = 2)
        ps, st = Lux.setup(RNG, m)
        X = randn(RNG, Float32, 32, 8)
        Y, _ = m(X, ps, st)
        @test size(Y) == (32, 8)
        @test all(isfinite, Y)
    end

    @testset "repel (Quellcode 3.9)" begin
        xs = Float32[-1.4, -0.5, 0.0, 0.5, 1.4, 2.7]
        r = repel(xs)
        @test r == Float32[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        # Custom pivot
        r2 = repel(Float32[0.3, 0.9]; pivot = 0.5)
        @test r2 == Float32[0.0, 1.0]
        # Non-mutating
        @test xs == Float32[-1.4, -0.5, 0.0, 0.5, 1.4, 2.7]
    end

    @testset "loss (Quellcode 3.8) — finite and positive" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG, m)
        # Input must be in [0,1] for binary cross-entropy.
        a = rand(RNG, Float32, 16, 4)
        b = rand(RNG, Float32, 16, 4)
        c = rand(RNG, Float32, 16, 4)
        loss, _ = deep_kate_loss(m, ps, st, a, b, c)
        @test isfinite(loss)
        @test loss > 0
    end

    @testset "single-batch loss overload — reconstruction BCE only" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG, m)
        X = rand(RNG, Float32, 16, 4)
        loss4, _ = deep_kate_loss(m, ps, st, X)
        @test isfinite(loss4)
        @test loss4 > 0
        # The 4-arg form equals the `ce` term of the 6-arg form when
        # `a == b == c` and the targets are taken in test mode (so the
        # prev / succ mse vs −target reduces to a non-zero self-distance).
        # We only assert the 4-arg loss is strictly *smaller* than the
        # 6-arg loss on identical inputs, because 6-arg adds the two
        # non-negative temporal terms on top.
        loss6, _ = deep_kate_loss(m, ps, st, X, X, X)
        @test loss4 < loss6
    end

    @testset "single-batch loss — differentiable" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG, m)
        X = rand(RNG, Float32, 16, 3)
        g = Zygote.gradient(p -> first(deep_kate_loss(m, p, st, X)), ps)[1]
        @test g !== nothing
        @test any(!iszero, g.layer_10.weight)     # decoder output
    end

    @testset "loss is differentiable through the encoder+decoder" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG, m)
        a = rand(RNG, Float32, 16, 2)
        b = rand(RNG, Float32, 16, 2)
        c = rand(RNG, Float32, 16, 2)
        g = Zygote.gradient(p -> first(deep_kate_loss(m, p, st, a, b, c)), ps)[1]
        @test g !== nothing
        # Gradient reaches some parameter — at minimum the decoder's last
        # Dense, which is directly upstream of the cross-entropy signal.
        total = sum(sum(abs, getproperty(g, Symbol(:layer_, i)).weight)
                    for i in 1:10
                    if hasproperty(getproperty(g, Symbol(:layer_, i)), :weight))
        @test total > 0
        @test any(!iszero, g.layer_10.weight)
    end

    @testset "training-mode encoder differs from test-mode (dropout)" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG, m)
        encoder = Chain(values(m.layers)[1:latent_layer(m)]...)
        ps_enc = NamedTuple{(:layer_1,:layer_2,:layer_3,:layer_4,:layer_5)}(
            ntuple(i -> getproperty(ps, Symbol(:layer_, i)), 5))
        st_enc = NamedTuple{(:layer_1,:layer_2,:layer_3,:layer_4,:layer_5)}(
            ntuple(i -> getproperty(st, Symbol(:layer_, i)), 5))
        x = rand(RNG, Float32, 16, 4)
        y_train, _ = encoder(x, ps_enc, st_enc)
        y_test, _ = encoder(x, ps_enc, Lux.testmode(st_enc))
        # Dropout differs between modes, so the two outputs cannot be identical.
        @test !all(y_train .≈ y_test)
    end
end
