"""
    Anomaly.Sequence

Sequence-level anomaly detectors — plan 001 Stage E. Complements
[`Anomaly.Instance`] (which scores a *single* event by
reconstruction error) with three detectors that score an event in
the context of its neighbours:

- [`deeplog_topk_anomaly`] / [`deeplog_scan`] — DeepLog (Du et al.
  CCS 2017). Flag a position when the true next event is *not* in
  the predictor's top-`k` softmax. Works with any `seq_lstm`-shaped
  model (`(T, B) -> (vocab, B)` logits).
- [`masked_surprise`] — LogBERT-style (Guo et al. 2021) per-position
  surprise. Returns a `(T-1, B)` matrix of negative log-likelihoods;
  peaks localise an anomaly inside the sequence. A true LogBERT
  uses a bidirectional masked LM, but the per-position NLL is the
  comparable signal under the AR predictor we already train.
- [`sequence_perplexity`] — SeqTransformer-style (e.g. Ott et al.
  2024 DetAD) per-sequence perplexity, `exp(mean(NLL))`. The usual
  Transformer-LM anomaly score; computed here with the same
  autoregressive NLL so the API is model-agnostic — any next-event
  predictor produced by [`SeqLSTM.seq_lstm`] (LSTM, peephole, or
  bidirectional) works unchanged.

All three take `(model, ps, st, sequence)` where `sequence` is a
`(T, B)` matrix of 1-based event ids (the same convention as
[`SeqLSTM.seq_lstm_loss`]). State is passed through `Lux.testmode`
so dropout stays off.
"""
module Sequence

using Lux
using NNlib: logsoftmax
using Statistics

export deeplog_topk_anomaly, deeplog_scan, masked_surprise, sequence_perplexity

# ---------------------------------------------------------------------------
# DeepLog (Du et al. CCS 2017)
# ---------------------------------------------------------------------------

"""
    deeplog_topk_anomaly(model, ps, st, sequence; k = 9) -> Vector{Bool}

Given a `(T, B)` integer-id sequence, run the predictor on
`sequence[1:T-1, :]` and return `Bool` per column: `true` when the
actual `sequence[T, :]` event is *not* among the top-`k` softmax
candidates. DeepLog's core anomaly rule.

`k` is clamped to the model's vocabulary size.
"""
function deeplog_topk_anomaly(model, ps, st,
                              sequence::AbstractMatrix{<:Integer};
                              k::Integer = 9)
    T, B = size(sequence)
    T >= 2 || throw(ArgumentError("sequence must have at least 2 time steps"))
    inputs = @view sequence[1:T-1, :]
    targets = @view sequence[T, :]
    logits, _ = model(inputs, ps, Lux.testmode(st))
    V = size(logits, 1)
    kk = min(Int(k), V)
    out = Vector{Bool}(undef, B)
    @inbounds for b in 1:B
        col = collect(@view logits[:, b])
        top = partialsortperm(col, 1:kk; rev = true)
        out[b] = !(Int(targets[b]) in top)
    end
    return out
end

"""
    deeplog_scan(model, ps, st, sequence; window = 10, k = 9)
        -> Vector{Bool}

Slide a length-`window` context over a 1-D `sequence` (`Vector{Int}`)
and return the per-position DeepLog flag for each step where a
prediction is possible. Output length is `length(sequence) - window`.

The function batches every window into one forward pass — fast on
long sequences, with the usual memory / length trade-off.
"""
function deeplog_scan(model, ps, st,
                      sequence::AbstractVector{<:Integer};
                      window::Integer = 10, k::Integer = 9)
    n = length(sequence)
    w = Int(window)
    n > w || return Bool[]
    B = n - w
    mat = Matrix{Int}(undef, w + 1, B)
    @inbounds for b in 1:B
        @views mat[:, b] .= sequence[b:b + w]
    end
    return deeplog_topk_anomaly(model, ps, st, mat; k = k)
end

# ---------------------------------------------------------------------------
# LogBERT-style per-position surprise
# ---------------------------------------------------------------------------

"""
    masked_surprise(model, ps, st, sequence) -> Matrix{Float64}

Per-position negative log-likelihood of each event given its
prefix. Output shape is `(T - 1, B)` for a `(T, B)` input — one
score for each step `t ∈ 2:T`. Higher = model was more *surprised*
by the actual event.

Peaks localise anomalies inside a sequence; the LogBERT paper uses
a bidirectional masked LM, but under an AR predictor the per-step
NLL is the comparable signal.

Complexity is O(T) forward passes; fine for short sequences
(T ≲ a few hundred), because each prefix re-runs the LSTM. A
single-pass variant would need `Recurrence(...; return_sequence=true)`
plus a time-distributed head — not available out of the box on the
current `seq_lstm` topology, so we keep the loop.
"""
function masked_surprise(model, ps, st,
                         sequence::AbstractMatrix{<:Integer})
    T, B = size(sequence)
    T >= 2 || throw(ArgumentError("sequence must have at least 2 time steps"))
    out = Matrix{Float64}(undef, T - 1, B)
    st_test = Lux.testmode(st)
    @inbounds for t in 2:T
        inputs = @view sequence[1:t - 1, :]
        targets = @view sequence[t, :]
        logits, _ = model(inputs, ps, st_test)
        lp = logsoftmax(logits; dims = 1)
        for b in 1:B
            out[t - 1, b] = -Float64(lp[Int(targets[b]), b])
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# SeqTransformer-style perplexity
# ---------------------------------------------------------------------------

"""
    sequence_perplexity(model, ps, st, sequence) -> Vector{Float64}

Per-sequence perplexity — `exp(mean(masked_surprise))` down each
column. One scalar per batch element; strictly positive, equals
`1.0` only for a model that is perfectly confident and correct at
every step.
"""
function sequence_perplexity(model, ps, st,
                             sequence::AbstractMatrix{<:Integer})
    S = masked_surprise(model, ps, st, sequence)
    return exp.(vec(mean(S; dims = 1)))
end

end # module Sequence
