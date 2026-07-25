#
# AD regression tests — verify Zygote-computed gradients against
# finite differences as the numerical gold standard. Any future AD
# backend swap (Enzyme, Mooncake) that passes this file agrees with
# the math, not just with itself.
#
# The `@ignore_derivatives` in `KATE.KCompetetive` marks the sort as
# non-differentiable on purpose — which is mathematically correct
# (the sort is a step function) but means gradients *through* the
# competition look subtly different from a literal FD at points close
# to the k-th winner boundary. We side-step that by testing AD only on
# paths that compose smooth ops (PeepholeLSTM, SeqLSTM, the KATE input
# away from the boundary, the DeepKATE objective at stable weights).
#

using Test
using FiniteDifferences: central_fdm, grad
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.KATE: KCompetetive
using LogClustering.DeepKATE: deep_kate, deep_kate_loss, latent_layer
using LogClustering.SeqLSTM: seq_lstm, seq_lstm_loss, PeepholeLSTM

const RNG_AD = Random.MersenneTwister(19)

"Hand-rolled scalar central-diff — cheaper than FiniteDifferences on small inputs."
function central_diff_grad(f, x::AbstractArray; h = 1e-4)
    g = similar(x, Float64)
    x_f64 = Float64.(x)
    @inbounds for i in eachindex(x_f64)
        old = x_f64[i]
        x_f64[i] = old + h
        fp = f(eltype(x).(x_f64))
        x_f64[i] = old - h
        fm = f(eltype(x).(x_f64))
        x_f64[i] = old
        g[i] = (fp - fm) / (2h)
    end
    return g
end

@testset "AD regression" begin
    @testset "KCompetetive — gradient w.r.t. input matches FD" begin
        # All-positive, magnitude-separated input so the k-winner boundary
        # is well-defined and small FD perturbations don't cross it.
        l = KCompetetive(6, 6, tanh; k = 4)
        ps, st = Lux.setup(RNG_AD, l)
        x = Float32[5.0, 3.0, 1.0, 0.8, 0.5, 0.3]
        loss(xx) = sum(first(l(xx, ps, st)))
        g_zyg = Zygote.gradient(loss, x)[1]
        g_fd  = central_diff_grad(loss, x; h = 1e-3)
        @test isapprox(g_zyg, g_fd; rtol = 1e-2, atol = 1e-3)
    end

    @testset "KCompetetive — gradient w.r.t. weight matches FD" begin
        l = KCompetetive(5, 4, tanh; k = 4)
        ps, st = Lux.setup(RNG_AD, l)
        x = Float32[1.0, 0.5, -0.5, -1.0, 0.2]
        loss(w) = sum(first(l(x, (weight = w, bias = ps.bias), st)))
        g_zyg = Zygote.gradient(loss, ps.weight)[1]
        g_fd  = central_diff_grad(loss, ps.weight; h = 1e-3)
        @test isapprox(g_zyg, g_fd; rtol = 5e-2, atol = 1e-3)
    end

    @testset "PeepholeLSTM — gradient w.r.t. input matches FD (3-D tensor)" begin
        l = PeepholeLSTM(3 => 4)
        ps, st = Lux.setup(RNG_AD, l)
        x = Float32.(0.1 .* randn(RNG_AD, 3, 4, 2))    # small values keep sigmoid linear-ish
        loss(xx) = sum(first(l(xx, ps, st)))
        g_zyg = Zygote.gradient(loss, x)[1]
        g_fd  = central_diff_grad(loss, x; h = 1e-3)
        @test isapprox(g_zyg, g_fd; rtol = 5e-2, atol = 1e-3)
    end

    @testset "PeepholeLSTM — gradient w.r.t. peephole weights matches FD" begin
        l = PeepholeLSTM(3 => 4)
        ps, st = Lux.setup(RNG_AD, l)
        x = Float32.(0.1 .* randn(RNG_AD, 3, 4, 2))
        loss(pi) = sum(first(l(x,
            merge(ps, (peep_i = pi,)), st)))
        g_zyg = Zygote.gradient(loss, ps.peep_i)[1]
        g_fd  = central_diff_grad(loss, ps.peep_i; h = 1e-3)
        @test isapprox(g_zyg, g_fd; rtol = 5e-2, atol = 1e-3)
    end

    @testset "seq_lstm — embedding gradient matches FD (tiny vocab)" begin
        m = seq_lstm(4; embed = 3, hidden = 4)
        ps, st = Lux.setup(RNG_AD, m)
        seq = Int[1 2; 2 3; 3 4]
        loss(W) = first(seq_lstm_loss(m,
            merge(ps, (layer_1 = (weight = W,),)),
            st, seq))
        g_zyg = Zygote.gradient(loss, ps.layer_1.weight)[1]
        g_fd  = central_diff_grad(loss, ps.layer_1.weight; h = 1e-3)
        @test isapprox(g_zyg, g_fd; rtol = 5e-2, atol = 1e-2)
    end

    @testset "deep_kate_loss — gradient w.r.t. target matches FD (stop-gradient test)" begin
        # `target_a` and `target_c` are wrapped in @ignore_derivatives
        # inside the loss (matches the thesis `Flux.data(...)` stop-
        # gradient). So Zygote must see zero gradient through them, and
        # FD must agree within perturbation noise.
        m = deep_kate(6; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_AD, m)
        b = rand(RNG_AD, Float32, 6, 2)
        c = rand(RNG_AD, Float32, 6, 2)
        loss(a) = first(deep_kate_loss(m, ps, Lux.testmode(st), a, b, c))
        a = rand(RNG_AD, Float32, 6, 2)
        g_zyg = Zygote.gradient(loss, a)[1]
        # Stop-gradient means Zygote returns `nothing`; treat it as zero
        # and confirm FD is also negligibly small (bounded by the
        # perturbation's interaction with dropout/RNG, which we nulled
        # out by running in testmode).
        if g_zyg === nothing
            g_zyg = zeros(Float32, size(a))
        end
        g_fd  = central_diff_grad(loss, a; h = 1e-3)
        @test maximum(abs, g_zyg) <= 1e-5
        @test maximum(abs, g_fd) <= 5e-3
    end
end

@testset "AD backend seam — DifferentiationInterface" begin
    # Sanity-check that DI can drive Zygote the same way the tests
    # above use Zygote directly. This is the seam for later swapping
    # to Enzyme / Mooncake without rewriting every call site.
    using DifferentiationInterface: AutoZygote, gradient
    f(x) = sum(abs2, x)
    x = Float32[1.0, -2.0, 3.0]
    g_di = gradient(f, AutoZygote(), x)
    g_an = 2 .* x
    @test isapprox(g_di, g_an; rtol = 1e-6)
end
