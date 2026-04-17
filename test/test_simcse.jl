using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.SimCSE: simcse_loss, simcse_step_loss

const RNG_SC = Random.MersenneTwister(31)

@testset "SimCSE" begin
    @testset "simcse_loss — identical views equal log(batch)" begin
        # If h⁺ ≡ h, every sample's positive is its best match. The
        # remaining off-diagonal distractors still contribute to the
        # softmax denominator, so at `τ → 0` the loss approaches 0;
        # at the default `τ = 0.05` the similarity spread between
        # diagonal and off-diagonal is huge, so the loss stays
        # small. We assert it's strictly less than the uniform
        # log(batch) baseline.
        rng = Random.MersenneTwister(0)
        h = randn(rng, Float32, 8, 16)
        l_id = simcse_loss(h, h)
        l_uniform = log(Float32(16))
        @test 0 <= l_id < l_uniform

        # Orthogonal views (negate every other column's sign): the
        # positive is no longer the diagonal, so loss rises.
        flipped = copy(h)
        flipped[:, 2:2:end] .*= -1
        l_flipped = simcse_loss(h, flipped)
        @test l_flipped > l_id
    end

    @testset "simcse_loss — argument validation" begin
        @test_throws DimensionMismatch simcse_loss(randn(3, 4), randn(3, 3))
        @test_throws ArgumentError    simcse_loss(randn(3, 1), randn(3, 1))
    end

    @testset "simcse_step_loss — differentiable through the encoder" begin
        m = Chain(Dense(4 => 8, tanh), Dropout(0.3), Dense(8 => 4))
        ps, st = Lux.setup(RNG_SC, m)
        x = randn(RNG_SC, Float32, 4, 6)
        loss, _ = simcse_step_loss(m, ps, st, x)
        @test isfinite(loss)
        @test loss > 0
        g = Zygote.gradient(p -> first(simcse_step_loss(m, p, st, x)), ps)[1]
        @test g !== nothing
        # At least one parameter in the encoder receives a non-zero
        # gradient — Dropout's stochastic mask is what makes the
        # positive-pair signal non-constant, so the gradient must
        # reach through it.
        @test any(!iszero, g.layer_1.weight)
    end

    @testset "SGD on a Dropout-wrapped encoder drives the loss down" begin
        # Tiny batch repeated many times: on a fixed input the two
        # forward passes diverge only via Dropout; SGD should push
        # the encoder to produce mask-invariant embeddings and the
        # SimCSE loss downward.
        m = Chain(Dense(6 => 8, tanh), Dropout(0.3), Dense(8 => 6))
        rng = Random.MersenneTwister(42)
        ps, st = Lux.setup(rng, m)
        x = randn(rng, Float32, 6, 8)

        function sgd!(ps, grads, lr)
            grads === nothing && return ps
            if ps isa AbstractArray
                return ps .- lr .* grads
            elseif ps isa NamedTuple
                ks = keys(ps)
                return NamedTuple{ks}(map(k -> sgd!(getfield(ps, k),
                                                    hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                    lr), ks))
            else
                return ps
            end
        end

        initial_loss, _ = simcse_step_loss(m, ps, st, x)
        for _ in 1:80
            (_, _), back = Zygote.pullback(p -> simcse_step_loss(m, p, st, x), ps)
            g = back((one(initial_loss), nothing))[1]
            ps = sgd!(ps, g, Float32(0.1))
        end
        final_loss, _ = simcse_step_loss(m, ps, st, x)
        @test final_loss < initial_loss
    end
end
