using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.SoftKATE: GumbelSoftCompetetive, soft_kate, soft_kate_loss,
                              set_temperature, anneal_temperature
using LogClustering.SoftKATE: latent_layer as sk_latent_layer

const RNG_SK = Random.MersenneTwister(101)

function _sgd_step_sk(ps, grads, lr)
    grads === nothing && return ps
    ps isa AbstractArray && return ps .- lr .* grads
    if ps isa NamedTuple
        k = keys(ps)
        return NamedTuple{k}(map(ki -> _sgd_step_sk(getfield(ps, ki),
                                                    hasproperty(grads, ki) ? getfield(grads, ki) : nothing,
                                                    lr), k))
    end
    return ps
end

@testset "SoftKATE" begin
    @testset "GumbelSoftCompetetive — shape + parameter layout" begin
        l = GumbelSoftCompetetive(8 => 4; init_temperature = 0.5f0)
        ps, st = Lux.setup(RNG_SK, l)
        @test size(ps.weight) == (4, 8)
        @test size(ps.bias)   == (4,)
        @test st.temperature  == 0.5f0
        # Forward: (in, batch) → (out, batch).
        x = randn(RNG_SK, Float32, 8, 3)
        y, _ = l(x, ps, st)
        @test size(y) == (4, 3)
        @test all(isfinite, y)
    end

    @testset "GumbelSoftCompetetive — lower temperature sparsifies the gate" begin
        # Feed a deterministic input and peek at the per-output magnitudes.
        # At τ ≈ 0, the output concentrates mass on one neuron per column;
        # at τ ≈ ∞ it spreads. Proxy: max/mean ratio per column.
        using Statistics: mean
        l = GumbelSoftCompetetive(4 => 8; init_temperature = 10.0f0)
        ps, st_hot = Lux.setup(RNG_SK, l)                   # τ = 10
        x = randn(RNG_SK, Float32, 4, 6)
        st_cold = merge(st_hot, (temperature = 0.05f0,))
        y_hot,  _ = l(x, ps, st_hot)
        y_cold, _ = l(x, ps, st_cold)
        ratio(y) = mean(maximum(abs.(y); dims = 1) ./
                         (mean(abs.(y); dims = 1) .+ 1f-8))
        @test ratio(y_cold) > ratio(y_hot)
    end

    @testset "soft_kate — parametric factory topology" begin
        m = soft_kate(32; hidden = [16, 8], latent = 4)
        # Encoder: [Dense, LN, Dropout] × 2 + GumbelSoftComp + Dense(sin)
        # Decoder: [Dense, LN, Dropout] × 2 + Dense(out)
        # Total: 6 + 1 + 1 + 6 + 1 = 15 (when dropout > 0, layernorm on)
        @test length(m.layers) == 15
        @test sk_latent_layer(m) isa Int
        # The tanh-activated Dense right after GumbelSoftCompetetive is the
        # bottleneck output.
        bottleneck = m.layers[sk_latent_layer(m)]
        @test bottleneck isa Dense
        @test bottleneck.activation === tanh
        # Decoder output is sigmoid so BCE is well-defined.
        @test m.layers[end] isa Dense
        @test m.layers[end].activation === sigmoid
        @test m.layers[end].out_dims == 32
    end

    @testset "soft_kate — forward shape preserved" begin
        m = soft_kate(24; hidden = [32, 16], latent = 8)
        ps, st = Lux.setup(RNG_SK, m)
        x = rand(RNG_SK, Float32, 24, 5)
        y, _ = m(x, ps, st)
        @test size(y) == (24, 5)
        @test all(0 .<= y .<= 1)    # sigmoid output
    end

    @testset "soft_kate_loss — finite, non-negative, differentiable" begin
        m = soft_kate(16; hidden = [16, 8], latent = 4)
        ps, st = Lux.setup(RNG_SK, m)
        x = rand(RNG_SK, Float32, 16, 4)
        loss, _ = soft_kate_loss(m, ps, st, x; λ = 0.5)
        @test loss isa Real
        @test isfinite(loss)
        @test loss > 0
        g = Zygote.gradient(
            p -> first(soft_kate_loss(m, p, st, x; λ = 0.5)), ps)[1]
        @test g !== nothing
        # Gradient reaches the bottleneck competition + decoder output.
        @test any(!iszero, g.layer_7.weight)   # GumbelSoftCompetetive
        last_layer = Symbol(:layer_, length(m.layers))
        @test any(!iszero, getproperty(g, last_layer).weight)
    end

    @testset "soft_kate_loss — λ = 0 ≡ reconstruction only; λ > 0 adds signal" begin
        m = soft_kate(16; hidden = [16, 8], latent = 4)
        ps, st = Lux.setup(RNG_SK, m)
        x = rand(RNG_SK, Float32, 16, 4)
        loss0, _ = soft_kate_loss(m, ps, st, x; λ = 0.0)
        loss1, _ = soft_kate_loss(m, ps, st, x; λ = 1.0)
        @test loss1 >= loss0        # extra non-negative contrastive term
    end

    @testset "anneal_temperature — cosine schedule 1.0 → 0.1 is monotone" begin
        m = soft_kate(8; hidden = [8, 4], latent = 2)
        _, st = Lux.setup(RNG_SK, m)
        τs = Float64[]
        for t in 1:8
            _, τ = anneal_temperature(st, t, 8;
                                      start = 1.0, stop = 0.1,
                                      schedule = :cosine)
            push!(τs, τ)
        end
        @test τs[1] ≈ 1.0 atol = 1e-6
        @test τs[end] ≈ 0.1 atol = 1e-6
        @test issorted(τs; rev = true)
    end

    @testset "set_temperature propagates to the GumbelSoftCompetetive sub-state" begin
        m = soft_kate(8; hidden = [8], latent = 2)
        _, st = Lux.setup(RNG_SK, m)
        st2 = set_temperature(st, 0.07)
        # Find the GumbelSoftCompetetive sub-state and assert τ landed.
        for k in keys(st2)
            sub = getfield(st2, k)
            hasproperty(sub, :temperature) || continue
            @test sub.temperature ≈ 0.07f0 atol = 1e-7
        end
    end

    @testset "anneal_temperature — linear + unknown-schedule error" begin
        m = soft_kate(8; hidden = [8], latent = 2)
        _, st = Lux.setup(RNG_SK, m)
        _, τmid = anneal_temperature(st, 5, 9;
                                     start = 1.0, stop = 0.0,
                                     schedule = :linear)
        @test τmid ≈ 0.5 atol = 1e-6
        @test_throws ArgumentError anneal_temperature(st, 1, 5;
                                                      schedule = :bogus)
    end

    @testset "soft_kate trains: loss decreases on a repeating fixture" begin
        m = soft_kate(8; hidden = [16, 8], latent = 4,
                      init_temperature = 1.0f0)
        ps, st = Lux.setup(RNG_SK, m)
        # Two repeating patterns — the AE has something to compress onto.
        x = hcat(
            repeat(Float32[0.1, 0.9, 0.0, 0.5, 0.0, 0.8, 0.0, 0.2], 1, 6),
            repeat(Float32[0.0, 0.2, 0.9, 0.1, 0.7, 0.0, 0.3, 0.0], 1, 6),
        )
        losses = Float64[]
        for step in 1:40
            st, _ = anneal_temperature(st, step, 40)
            (loss, st), back = Zygote.pullback(
                p -> soft_kate_loss(m, p, st, x; λ = 0.3), ps)
            g = back((one(loss), nothing))[1]
            ps = _sgd_step_sk(ps, g, 0.05f0)
            push!(losses, Float64(loss))
        end
        @test losses[end] < losses[1]
    end
end
