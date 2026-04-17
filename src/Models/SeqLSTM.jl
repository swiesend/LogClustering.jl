"""
    SeqLSTM

Port of the thesis's next-event predictor (§3.2.8, *Eventvoraussage*) to
Lux. The original model used Flux.jl LSTMs wrapped in a custom
`Parallel` struct to build bi-directional stacks; Lux provides both
[`Lux.Recurrence`](@ref) and [`Lux.BidirectionalRNN`](@ref) natively, so
the port is a thin layer.

`seq_lstm(vocab_size; hidden, bidirectional)` returns a Lux `Chain`
whose forward takes a `(sequence_length, batch)` matrix of 1-based
event ids and emits `(vocab_size, batch)` unnormalised logits over the
next event.

Training / prediction helpers:

- [`seq_lstm_loss`] — cross-entropy between the final logits of a
  length-`T` sliced input `(1:T-1, :)` and the last-event target
  `sequence[T, :]`. Matches the thesis's "predict the next event in a
  window" objective.
- [`predict_next`] — argmax of the softmax over the logits (useful for
  qualitative inspection; stochastic sampling is trivially expressed
  on top of the logits via `softmax`).

Peephole LSTM (Gers & Schmidhuber 2000, thesis ref [22]) is still TODO
— Lux's stock `LSTMCell` is the plain Hochreiter & Schmidhuber 1997
variant, which is what the thesis compared against.
"""
module SeqLSTM

using Lux
using Random
using Statistics
using NNlib: logsoftmax

export seq_lstm, seq_lstm_loss, predict_next

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

"""
    seq_lstm(vocab_size; embed = 32, hidden = 64) -> Chain

Build a next-event predictor for a vocabulary of `vocab_size` distinct
events. Input: `(sequence_length, batch)` integer matrix of 1-based ids.
Output: `(vocab_size, batch)` logits over the next event.

`embed` is the embedding dimension, `hidden` the LSTM hidden size.

A bi-directional variant using [`Lux.BidirectionalRNN`](@ref) is still
open — Lux's `BidirectionalRNN` returns a 3-D `(features, seq, batch)`
tensor that needs a pooling step before the projection head. That
plumbing plus the thesis's peephole LSTM (Gers & Schmidhuber 2000,
thesis ref [22]) will land in a follow-up.
"""
function seq_lstm(vocab_size::Integer;
                  embed::Integer = 32,
                  hidden::Integer = 64)
    return Chain(
        Embedding(vocab_size => embed),
        Recurrence(LSTMCell(embed => hidden)),
        Dense(hidden => vocab_size),
    )
end

# ---------------------------------------------------------------------------
# Loss and prediction
# ---------------------------------------------------------------------------

"""
    seq_lstm_loss(model, ps, st, sequence) -> (loss, st_new)

Negative log-likelihood of `sequence[end, :]` under the model conditioned
on `sequence[1:end-1, :]`. `sequence` is a `(T, batch)` integer matrix;
the function slices off the last row, runs the model, and scores the
cross-entropy against that row.
"""
function seq_lstm_loss(model, ps, st, sequence::AbstractMatrix{<:Integer})
    T, B = size(sequence)
    T >= 2 || throw(ArgumentError("sequence must have at least 2 time steps"))
    inputs = @view sequence[1:T-1, :]
    targets = @view sequence[T, :]
    logits, st_new = model(inputs, ps, st)
    lp = logsoftmax(logits; dims = 1)
    # Gather log-probabilities of the correct targets.
    total = zero(eltype(lp))
    @inbounds for b in 1:B
        total -= lp[Int(targets[b]), b]
    end
    return total / B, st_new
end

"""
    predict_next(model, ps, st, sequence) -> Vector{Int}

Greedy next-event prediction: argmax of the softmax over each column of
the model's logits. `sequence` is a `(T, batch)` matrix of history;
returns a length-`batch` vector of predicted event ids.
"""
function predict_next(model, ps, st, sequence::AbstractMatrix{<:Integer})
    logits, _ = model(sequence, ps, Lux.testmode(st))
    B = size(logits, 2)
    out = Vector{Int}(undef, B)
    @inbounds for b in 1:B
        # argmax over vocabulary dimension
        best_i = 1
        best_v = logits[1, b]
        for i in 2:size(logits, 1)
            v = logits[i, b]
            v > best_v && (best_v = v; best_i = i)
        end
        out[b] = best_i
    end
    return out
end

end # module SeqLSTM
