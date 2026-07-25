using Test
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.SeqLSTM: seq_lstm, seq_lstm_loss
using LogClustering.Sequence: deeplog_topk_anomaly, deeplog_scan,
                              masked_surprise, sequence_perplexity

const RNG_ANOM_SEQ = Random.MersenneTwister(19)

# Re-use the tiny SGD helper idiom from test_seqlstm.
function _sgd_step_seq(ps, grads, lr)
    if grads === nothing || grads isa Nothing
        return ps
    elseif ps isa AbstractArray
        return ps .- lr .* grads
    elseif ps isa NamedTuple
        keys_ = keys(ps)
        return NamedTuple{keys_}(map(k -> _sgd_step_seq(getfield(ps, k),
                                                       hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                       lr), keys_))
    else
        return ps
    end
end

# Train a tiny LSTM on the 2-cycle `[1, 2, 1, 2, ...]` so the predictor
# learns: after `[1, 2, 1]` the correct next event is `2`, and after
# `[2, 1, 2]` it is `1`. The vocabulary includes event `3` which is
# never seen during training, so it stays low-probability and serves
# as the "unknown" anomaly target.
function _train_cycle_model(; steps = 400, lr = 0.05f0)
    m = seq_lstm(3; embed = 4, hidden = 8)
    ps, st = Lux.setup(RNG_ANOM_SEQ, m)
    # Two training sequences cover both phase offsets of the cycle.
    training = [
        reshape([1, 2, 1, 2], :, 1),   # inputs [1,2,1] → target 2
        reshape([2, 1, 2, 1], :, 1),   # inputs [2,1,2] → target 1
    ]
    for step in 1:steps
        seq = training[mod1(step, length(training))]
        (_, _), back = Zygote.pullback(p -> seq_lstm_loss(m, p, st, seq), ps)
        g = back((one(Float32), nothing))[1]
        ps = _sgd_step_seq(ps, g, lr)
    end
    return m, ps, st
end

@testset "Anomaly.Sequence" begin
    m, ps, st = _train_cycle_model()

    @testset "deeplog_topk_anomaly — shape + validity" begin
        seq = reshape([1, 2, 1, 2], :, 1)
        flags = deeplog_topk_anomaly(m, ps, st, seq; k = 1)
        @test length(flags) == 1
        @test flags isa Vector{Bool}
    end

    @testset "deeplog_topk_anomaly — k = vocab catches nothing" begin
        seq = reshape([1, 2, 1, 2], :, 1)
        flags = deeplog_topk_anomaly(m, ps, st, seq; k = 3)
        @test flags == [false]
    end

    @testset "deeplog_topk_anomaly — k = 1 separates trained vs anomalous target" begin
        # Trained model should predict `2` after `1, 2, 1` — target `2` = OK,
        # target `3` = anomalous.
        normal_seq   = reshape([1, 2, 1, 2], :, 1)
        anomaly_seq  = reshape([1, 2, 1, 3], :, 1)
        @test deeplog_topk_anomaly(m, ps, st, normal_seq; k = 1) == [false]
        @test deeplog_topk_anomaly(m, ps, st, anomaly_seq; k = 1) == [true]
    end

    @testset "deeplog_topk_anomaly — sequences shorter than 2 rejected" begin
        @test_throws ArgumentError deeplog_topk_anomaly(
            m, ps, st, reshape([1], :, 1))
    end

    @testset "deeplog_topk_anomaly — handles batch dimension" begin
        batch = hcat([1, 2, 1, 2], [1, 2, 1, 3])   # col 1 normal, col 2 anomaly
        flags = deeplog_topk_anomaly(m, ps, st, batch; k = 1)
        @test length(flags) == 2
        @test flags[1] == false
        @test flags[2] == true
    end

    @testset "deeplog_scan — sliding window over a long sequence" begin
        # `[1,2,1,2,1,2,3,2]` with window=3:
        #   windows 1..5 each predict position 4..8.
        #   Only position 7 (value 3 after 1,2,1) should flag as k=1 anomaly.
        long = [1, 2, 1, 2, 1, 2, 3, 2]
        flags = deeplog_scan(m, ps, st, long; window = 3, k = 1)
        @test length(flags) == length(long) - 3
        # The offending transition lands inside the window ending at pos 7.
        @test any(flags)
    end

    @testset "deeplog_scan — empty when sequence too short" begin
        @test deeplog_scan(m, ps, st, [1, 2]; window = 5) == Bool[]
    end

    @testset "masked_surprise — shape + non-negative" begin
        seq = reshape([1, 2, 1, 2], :, 1)
        S = masked_surprise(m, ps, st, seq)
        @test size(S) == (3, 1)
        @test all(S .>= 0)
    end

    @testset "masked_surprise — anomalous step has higher NLL than normal" begin
        normal_seq  = reshape([1, 2, 1, 2], :, 1)
        anomaly_seq = reshape([1, 2, 1, 3], :, 1)
        @test masked_surprise(m, ps, st, normal_seq)[end, 1] <
              masked_surprise(m, ps, st, anomaly_seq)[end, 1]
    end

    @testset "masked_surprise — batched is per-column independent" begin
        batch = hcat([1, 2, 1, 2], [1, 2, 1, 3])
        S = masked_surprise(m, ps, st, batch)
        @test size(S) == (3, 2)
        @test S[end, 1] < S[end, 2]
    end

    @testset "masked_surprise — sequences shorter than 2 rejected" begin
        @test_throws ArgumentError masked_surprise(
            m, ps, st, reshape([1], :, 1))
    end

    @testset "sequence_perplexity — positive per-column" begin
        batch = hcat([1, 2, 1, 2], [1, 2, 1, 3])
        ppl = sequence_perplexity(m, ps, st, batch)
        @test length(ppl) == 2
        @test all(ppl .>= 1.0)         # perplexity ≥ 1 by definition
        @test ppl[1] < ppl[2]          # the anomalous column has higher ppl
    end
end
