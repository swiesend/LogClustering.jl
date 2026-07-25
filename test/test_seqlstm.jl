using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.SeqLSTM: seq_lstm, seq_lstm_loss, predict_next, PeepholeLSTM

const RNG_SEQ = Random.MersenneTwister(11)

# Minimal recursive SGD update over a NamedTuple tree of params.
function _sgd_step(ps, grads, lr)
    if grads === nothing || grads isa Nothing
        return ps
    elseif ps isa AbstractArray
        return ps .- lr .* grads
    elseif ps isa NamedTuple
        keys_ = keys(ps)
        return NamedTuple{keys_}(map(k -> _sgd_step(getfield(ps, k),
                                                   hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                   lr), keys_))
    else
        return ps
    end
end

@testset "SeqLSTM" begin
    @testset "model shapes" begin
        m = seq_lstm(20; embed = 8, hidden = 16)
        ps, st = Lux.setup(RNG_SEQ, m)
        x = rand(RNG_SEQ, 1:20, 5, 3)
        y, _ = m(x, ps, st)
        @test size(y) == (20, 3)                        # (vocab, batch)
        @test eltype(y) == Float32
    end

    @testset "loss is a non-negative scalar" begin
        m = seq_lstm(12; embed = 6, hidden = 10)
        ps, st = Lux.setup(RNG_SEQ, m)
        seq = rand(RNG_SEQ, 1:12, 4, 3)
        loss, _ = seq_lstm_loss(m, ps, st, seq)
        @test loss isa Real
        @test loss >= 0
    end

    @testset "loss rejects single-step sequences" begin
        m = seq_lstm(10; embed = 4, hidden = 8)
        ps, st = Lux.setup(RNG_SEQ, m)
        @test_throws ArgumentError seq_lstm_loss(m, ps, st, rand(RNG_SEQ, 1:10, 1, 2))
    end

    @testset "loss is differentiable (Zygote)" begin
        m = seq_lstm(8; embed = 4, hidden = 6)
        ps, st = Lux.setup(RNG_SEQ, m)
        seq = rand(RNG_SEQ, 1:8, 3, 2)
        g = Zygote.gradient(p -> first(seq_lstm_loss(m, p, st, seq)), ps)[1]
        @test g !== nothing
        # At least one parameter receives non-zero gradient.
        total = sum(abs, g.layer_3.weight) + sum(abs, g.layer_2.weight_ih)
        @test total > 0
    end

    @testset "predict_next returns length-batch vector of valid ids" begin
        m = seq_lstm(9; embed = 4, hidden = 8)
        ps, st = Lux.setup(RNG_SEQ, m)
        seq = rand(RNG_SEQ, 1:9, 4, 5)
        pred = predict_next(m, ps, st, seq)
        @test length(pred) == 5
        @test all(1 .<= pred .<= 9)
    end

    @testset "model memorises a single tiny sequence after a few SGD steps" begin
        # One sequence, one batch element, train to convergence on a two-class
        # vocabulary. A few hundred steps of plain SGD should drop the loss well
        # below the 2-class uniform baseline of log(2) ≈ 0.693.
        m = seq_lstm(2; embed = 4, hidden = 8)
        ps, st = Lux.setup(RNG_SEQ, m)
        seq = reshape([1, 2, 1, 2, 1], :, 1)
        lr = Float32(0.05)
        final_loss = 0.0
        for _ in 1:200
            (loss, _), back = Zygote.pullback(p -> seq_lstm_loss(m, p, st, seq), ps)
            g = back((one(loss), nothing))[1]
            ps = _sgd_step(ps, g, lr)
            final_loss = loss
        end
        @test final_loss < log(2)
    end

    @testset "PeepholeLSTM layer — forward, parameter shapes, gradients" begin
        l = PeepholeLSTM(4 => 6)
        ps, st = Lux.setup(RNG_SEQ, l)
        @test size(ps.weight_i) == (24, 4)             # 4·out × in
        @test size(ps.weight_h) == (24, 6)             # 4·out × out
        @test size(ps.peep_i)   == (6,)
        @test size(ps.peep_f)   == (6,)
        @test size(ps.peep_o)   == (6,)
        @test size(ps.bias)     == (24,)

        x = randn(RNG_SEQ, Float32, 4, 7, 3)            # (in, T, batch)
        y, _ = l(x, ps, st)
        @test size(y) == (6, 3)                         # final hidden
        @test all(isfinite, y)

        g = Zygote.gradient(p -> sum(abs2, first(l(x, p, st))), ps)[1]
        @test g !== nothing
        @test any(!iszero, g.weight_i)
        @test any(!iszero, g.peep_i) || any(!iszero, g.peep_f) || any(!iszero, g.peep_o)
    end

    @testset "bidirectional seq_lstm — forward shape and differentiability" begin
        m = seq_lstm(8; embed = 4, hidden = 5, bidirectional = true)
        ps, st = Lux.setup(RNG_SEQ, m)
        x = rand(RNG_SEQ, 1:8, 4, 3)
        y, _ = m(x, ps, st)
        @test size(y) == (8, 3)
        seq = rand(RNG_SEQ, 1:8, 4, 2)
        g = Zygote.gradient(p -> first(seq_lstm_loss(m, p, st, seq)), ps)[1]
        @test g !== nothing
    end

    @testset "peephole seq_lstm — forward shape and differentiability" begin
        m = seq_lstm(8; embed = 4, hidden = 5, peephole = true)
        ps, st = Lux.setup(RNG_SEQ, m)
        x = rand(RNG_SEQ, 1:8, 4, 3)
        y, _ = m(x, ps, st)
        @test size(y) == (8, 3)
        seq = rand(RNG_SEQ, 1:8, 4, 2)
        g = Zygote.gradient(p -> first(seq_lstm_loss(m, p, st, seq)), ps)[1]
        @test g !== nothing
    end

    @testset "bidirectional + peephole — forward shape" begin
        m = seq_lstm(8; embed = 4, hidden = 5,
                     bidirectional = true, peephole = true)
        ps, st = Lux.setup(RNG_SEQ, m)
        x = rand(RNG_SEQ, 1:8, 4, 3)
        y, _ = m(x, ps, st)
        @test size(y) == (8, 3)
    end
end
