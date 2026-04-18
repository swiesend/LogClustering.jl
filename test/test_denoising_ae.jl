using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.DenoisingAE: denoising_ae, denoising_ae_loss

const RNG_DAE = Random.MersenneTwister(23)

# Generic SGD helper — same idiom as test_seqlstm.jl.
function _sgd_step_dae(ps, grads, lr)
    if grads === nothing || grads isa Nothing
        return ps
    elseif ps isa AbstractArray
        return ps .- lr .* grads
    elseif ps isa NamedTuple
        keys_ = keys(ps)
        return NamedTuple{keys_}(map(k -> _sgd_step_dae(getfield(ps, k),
                                                       hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                       lr), keys_))
    else
        return ps
    end
end

# Train for `steps` SGD iterations; return the trajectory of losses.
function _train_dae!(m, ps, st, x; steps = 50, lr = 0.05f0, kwargs...)
    losses = Float64[]
    for _ in 1:steps
        (loss, _), back = Zygote.pullback(
            p -> denoising_ae_loss(m, p, st, x; rng = RNG_DAE, kwargs...), ps)
        push!(losses, Float64(loss))
        g = back((one(Float32), nothing))[1]
        ps = _sgd_step_dae(ps, g, lr)
    end
    return ps, losses
end

@testset "DenoisingAE" begin
    @testset "construction + layer shapes" begin
        m = denoising_ae(16; hidden = 8, latent = 4)
        ps, st = Lux.setup(RNG_DAE, m)
        @test m isa Chain
        @test length(m.layers) == 4
        @test LogClustering.DenoisingAE.latent_layer(m) == 2
        # Encoder-in → encoder-out shapes.
        @test size(ps.layer_1.weight) == (8, 16)
        @test size(ps.layer_2.weight) == (4, 8)     # latent
        @test size(ps.layer_3.weight) == (8, 4)
        @test size(ps.layer_4.weight) == (16, 8)
    end

    @testset "forward shape is preserved" begin
        m = denoising_ae(16; hidden = 8, latent = 4)
        ps, st = Lux.setup(RNG_DAE, m)
        x = rand(RNG_DAE, Float32, 16, 5)
        y, _ = m(x, ps, st)
        @test size(y) == size(x)
        @test all(0 .<= y .<= 1)         # sigmoid output
    end

    @testset "loss is a non-negative scalar" begin
        m = denoising_ae(16; hidden = 8, latent = 4)
        ps, st = Lux.setup(RNG_DAE, m)
        x = rand(RNG_DAE, Float32, 16, 5)
        loss, _ = denoising_ae_loss(m, ps, st, x; mask_rate = 0.3, rng = RNG_DAE)
        @test loss isa Real
        @test loss >= 0
    end

    @testset "loss rejects out-of-range knobs" begin
        m = denoising_ae(4; hidden = 4, latent = 2)
        ps, st = Lux.setup(RNG_DAE, m)
        x = rand(RNG_DAE, Float32, 4, 2)
        @test_throws ArgumentError denoising_ae_loss(m, ps, st, x; mask_rate = 1.0)
        @test_throws ArgumentError denoising_ae_loss(m, ps, st, x; mask_rate = -0.1)
        @test_throws ArgumentError denoising_ae_loss(m, ps, st, x; σ = -1.0)
    end

    @testset "loss is differentiable (Zygote)" begin
        m = denoising_ae(8; hidden = 6, latent = 3)
        ps, st = Lux.setup(RNG_DAE, m)
        x = rand(RNG_DAE, Float32, 8, 4)
        g = Zygote.gradient(
            p -> first(denoising_ae_loss(m, p, st, x; rng = RNG_DAE)), ps)[1]
        @test g !== nothing
        total = sum(abs, g.layer_1.weight) +
                sum(abs, g.layer_2.weight) +
                sum(abs, g.layer_4.weight)
        @test total > 0
    end

    @testset "mask_rate = 0 recovers plain AE — loss drops on fixed fixture" begin
        m = denoising_ae(4; hidden = 4, latent = 2)
        ps, st = Lux.setup(RNG_DAE, m)
        x = Float32[0.1 0.2 0.3 0.4;
                    0.4 0.3 0.2 0.1;
                    0.1 0.1 0.1 0.1;
                    0.2 0.2 0.2 0.2]
        _, losses = _train_dae!(m, ps, st, x; steps = 80, lr = 0.1f0,
                                mask_rate = 0.0)
        @test losses[end] < losses[1]
    end

    @testset "mask_rate > 0 still trains" begin
        m = denoising_ae(4; hidden = 6, latent = 2)
        ps, st = Lux.setup(RNG_DAE, m)
        # Repeating pattern so a denoising objective has signal to learn.
        x = repeat(Float32[0.1, 0.9, 0.3, 0.7], 1, 6)
        _, losses = _train_dae!(m, ps, st, x; steps = 120, lr = 0.1f0,
                                mask_rate = 0.3)
        @test losses[end] < losses[1]
    end

    @testset "σ > 0 branch stays finite" begin
        m = denoising_ae(4; hidden = 4, latent = 2)
        ps, st = Lux.setup(RNG_DAE, m)
        x = rand(RNG_DAE, Float32, 4, 3)
        loss, _ = denoising_ae_loss(m, ps, st, x;
                                    mask_rate = 0.0, σ = 0.5, rng = RNG_DAE)
        @test isfinite(loss)
        @test loss >= 0
    end
end
