using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.DeepKATE
using LogClustering.DeepKATE: deep_kate, latent_layer, deep_kate_loss, repel

const RNG = Random.MersenneTwister(42)

@testset "DeepKATE" begin
    @testset "architecture — thesis defaults (hidden = [100, 20])" begin
        m = deep_kate(64; latent = 3)
        # Encoder: KComp + Dense + Dropout + KComp + Dense(sin) = 5
        # Decoder: Dense + Dense + Dropout + Dense          = 4
        @test length(m.layers) == 9
        @test latent_layer(m) == 5

        bottleneck = m.layers[latent_layer(m)]
        @test bottleneck isa Dense
        @test bottleneck.activation === sin
        @test bottleneck.in_dims == 3                    # latent
        @test bottleneck.out_dims == 3

        decoder_last = m.layers[end]
        @test decoder_last isa Dense
        @test decoder_last.activation === sigmoid
        @test decoder_last.out_dims == 64                # matches input

        # Two KCompetetive layers with the thesis's k values.
        ks = [l.k for l in m.layers if l isa LogClustering.KATE.KCompetetive]
        @test ks == [25, 3]
    end

    @testset "architecture — widened (hidden = [256, 128, 64], latent = 32)" begin
        m = deep_kate(512; hidden = [256, 128, 64], latent = 32, k1 = 64)
        # Encoder: KComp + (Dense+Dropout)×2 + KComp + Dense(sin) = 7
        # Decoder: Dense + (Dense+Dropout)×2 + Dense             = 6
        @test length(m.layers) == 13
        @test latent_layer(m) == 7

        bottleneck = m.layers[latent_layer(m)]
        @test bottleneck isa Dense
        @test bottleneck.activation === sin
        @test bottleneck.in_dims == 32
        @test bottleneck.out_dims == 32

        ks = [l.k for l in m.layers if l isa LogClustering.KATE.KCompetetive]
        @test ks == [64, 32]                             # k1 first, k_bottleneck second
    end

    @testset "architecture — minimal (hidden = [])" begin
        m = deep_kate(8; hidden = Int[], latent = 2, k1 = 2)
        # KComp(n→latent) + Dense(latent→latent, sin) + Dense(latent→n, sigmoid) = 3
        @test length(m.layers) == 3
        @test latent_layer(m) == 2

        ks = [l.k for l in m.layers if l isa LogClustering.KATE.KCompetetive]
        @test ks == [2]                                  # only bottleneck KComp

        decoder_last = m.layers[end]
        @test decoder_last.out_dims == 8
    end

    @testset "forward pass — widened bottleneck preserves shape" begin
        m = deep_kate(32; hidden = [64, 32], latent = 16, k1 = 16)
        ps, st = Lux.setup(RNG, m)
        X = randn(RNG, Float32, 32, 4)
        Y, _ = m(X, ps, st)
        @test size(Y) == (32, 4)
        @test all(isfinite, Y)
    end

    @testset "constructor rejects out-of-range knobs" begin
        @test_throws ArgumentError deep_kate(16; latent = 0)
        @test_throws ArgumentError deep_kate(16; k1 = 0)
        @test_throws ArgumentError deep_kate(16; k_bottleneck = 0)
        @test_throws ArgumentError deep_kate(16; hidden = [100, -1])
    end

    @testset "latent_layer rejects non-deep_kate chains" begin
        bogus = Chain(Dense(4 => 4, tanh))
        @test_throws ArgumentError latent_layer(bogus)
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
        last_layer = Symbol(:layer_, length(m.layers))
        @test any(!iszero, getproperty(g, last_layer).weight)   # decoder output
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
                    for i in 1:length(m.layers)
                    if hasproperty(getproperty(g, Symbol(:layer_, i)), :weight))
        @test total > 0
        last_layer = Symbol(:layer_, length(m.layers))
        @test any(!iszero, getproperty(g, last_layer).weight)
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
