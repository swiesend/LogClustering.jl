"""
    SeqLSTM

Port of the thesis's next-event predictor (§3.2.8, *Eventvoraussage*) to
Lux, with the thesis's bi-directional and peephole variants included.

- [`seq_lstm`] is the main entry point; `bidirectional=true` stacks two
  Lux `LSTMCell`s inside a `BidirectionalRNN` and pools the last time
  step, while `peephole=true` swaps the cell for [`PeepholeLSTM`], the
  Gers & Schmidhuber 2000 variant.
- [`PeepholeLSTM`] is a full-sequence layer (not a Lux `AbstractRecurrentCell`)
  implementing peephole gates `i`, `f`, `o` with per-unit diagonal
  weights `V_i`, `V_f`, `V_o` from the cell state.
- [`seq_lstm_loss`] — negative log-likelihood of `sequence[end, :]`
  given `sequence[1:end-1, :]`.
- [`predict_next`] — greedy argmax over the softmax of the logits.
"""
module SeqLSTM

using Lux
using LuxCore: LuxCore, AbstractLuxLayer
using Random
using Statistics
using WeightInitializers: glorot_uniform, zeros32
using NNlib: logsoftmax, sigmoid

export seq_lstm, seq_lstm_loss, predict_next, PeepholeLSTM

# ---------------------------------------------------------------------------
# Peephole LSTM (thesis ref [22]: Gers & Schmidhuber 2000)
# ---------------------------------------------------------------------------

"""
    PeepholeLSTM(in => out; init_weight = glorot_uniform,
                 init_peep = glorot_uniform, init_bias = zeros32)

Peephole LSTM (Gers & Schmidhuber 2000). Runs the recurrence internally
over the whole sequence and returns only the final hidden state, mirroring
`Lux.Recurrence(LSTMCell(...); return_sequence = false)`.

Gate equations:

```
i = σ(W_i·x + U_i·h + V_i ⊙ c_{t-1} + b_i)
f = σ(W_f·x + U_f·h + V_f ⊙ c_{t-1} + b_f)
c̃ = tanh(W_c·x + U_c·h + b_c)
c = f ⊙ c_{t-1} + i ⊙ c̃
o = σ(W_o·x + U_o·h + V_o ⊙ c + b_o)       (peephole on *current* c)
h = o ⊙ tanh(c)
```

Parameters: `weight_i` `(4·out, in)`, `weight_h` `(4·out, out)`,
`peep_i`/`peep_f`/`peep_o` `(out,)`, `bias` `(4·out,)`. Input shape
`(in, T, batch)`; output shape `(out, batch)`.
"""
struct PeepholeLSTM{IW, HW, PV, BI} <: AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    init_weight::IW
    init_peep::PV
    init_hidden::HW
    init_bias::BI
end

function PeepholeLSTM(
    pair::Pair{<:Integer, <:Integer};
    init_weight = glorot_uniform,
    init_peep = glorot_uniform,
    init_bias = zeros32,
)
    return PeepholeLSTM(Int(pair.first), Int(pair.second),
                        init_weight, init_peep, init_weight, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::PeepholeLSTM)
    H, I = l.out_dims, l.in_dims
    return (
        weight_i = l.init_weight(rng, 4H, I),
        weight_h = l.init_hidden(rng, 4H, H),
        peep_i   = l.init_peep(rng, H),
        peep_f   = l.init_peep(rng, H),
        peep_o   = l.init_peep(rng, H),
        bias     = l.init_bias(rng, 4H),
    )
end

LuxCore.initialstates(::AbstractRNG, ::PeepholeLSTM) = NamedTuple()
LuxCore.parameterlength(l::PeepholeLSTM) =
    4l.out_dims * (l.in_dims + l.out_dims + 1) + 3l.out_dims
LuxCore.statelength(::PeepholeLSTM) = 0

function Base.show(io::IO, l::PeepholeLSTM)
    print(io, "PeepholeLSTM(", l.in_dims, " => ", l.out_dims, ")")
end

function (l::PeepholeLSTM)(x::AbstractArray{T, 3}, ps, st::NamedTuple) where {T}
    H = l.out_dims
    _, seqlen, batch = size(x)
    h = zeros(T, H, batch)
    c = zeros(T, H, batch)
    @inbounds for t in 1:seqlen
        xt = @view x[:, t, :]
        z = ps.weight_i * xt .+ ps.weight_h * h .+ ps.bias
        zi = @view z[1:H, :]
        zf = @view z[H+1:2H, :]
        zc = @view z[2H+1:3H, :]
        zo = @view z[3H+1:4H, :]
        i = sigmoid.(zi .+ ps.peep_i .* c)
        f = sigmoid.(zf .+ ps.peep_f .* c)
        c = f .* c .+ i .* tanh.(zc)
        o = sigmoid.(zo .+ ps.peep_o .* c)
        h = o .* tanh.(c)
    end
    return h, st
end

# ---------------------------------------------------------------------------
# LastTimeStep — tiny pool to collapse (features, T, batch) to (features, batch)
# ---------------------------------------------------------------------------

"""
    LastTimeStep()

Return the last element of a `Vector{Matrix}` time sequence (Lux's
`BidirectionalRNN` output) or the last time slice of a 3-D tensor. Used
as the pool between a bidirectional recurrence and the projection head.
"""
struct LastTimeStep <: AbstractLuxLayer end

LuxCore.initialparameters(::AbstractRNG, ::LastTimeStep) = NamedTuple()
LuxCore.initialstates(::AbstractRNG, ::LastTimeStep) = NamedTuple()
LuxCore.parameterlength(::LastTimeStep) = 0
LuxCore.statelength(::LastTimeStep) = 0

(l::LastTimeStep)(xs::AbstractVector{<:AbstractMatrix}, _, st) = (xs[end], st)
(l::LastTimeStep)(xs::AbstractArray{<:Any, 3}, _, st) = (xs[:, end, :], st)

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

"""
    seq_lstm(vocab_size; embed = 32, hidden = 64,
             bidirectional = false, peephole = false) -> Chain

Build a next-event predictor. Input: `(sequence_length, batch)` integer
matrix of 1-based ids. Output: `(vocab_size, batch)` logits.

- `bidirectional = true` — stack a Lux `BidirectionalRNN(LSTMCell)` and
  pool the last time step before the projection (`2·hidden → vocab`).
- `peephole = true` — swap the stock `LSTMCell` for [`PeepholeLSTM`]
  (Gers & Schmidhuber 2000).
- The two flags are orthogonal; enabling both gives a bidirectional
  peephole LSTM.
"""
function seq_lstm(vocab_size::Integer;
                  embed::Integer = 32,
                  hidden::Integer = 64,
                  bidirectional::Bool = false,
                  peephole::Bool = false)
    rnn, proj_in = _rnn_block(Int(embed), Int(hidden), bidirectional, peephole)
    return Chain(
        Embedding(vocab_size => embed),
        rnn,
        Dense(proj_in => vocab_size),
    )
end

function _rnn_block(embed::Int, hidden::Int, bidirectional::Bool, peephole::Bool)
    if bidirectional && peephole
        # Two independent peephole passes — forward and on a reversed copy.
        # LastTimeStep on each, then concat over features.
        return (Parallel(
            vcat,
            forward  = PeepholeLSTM(embed => hidden),
            backward = Chain(
                ReverseSequence(2),
                PeepholeLSTM(embed => hidden),
            ),
        ), 2 * hidden)
    elseif bidirectional
        return (Chain(
            BidirectionalRNN(LSTMCell(embed => hidden)),
            LastTimeStep(),
        ), 2 * hidden)
    elseif peephole
        return (PeepholeLSTM(embed => hidden), hidden)
    else
        return (Recurrence(LSTMCell(embed => hidden)), hidden)
    end
end

# ---------------------------------------------------------------------------
# Loss and prediction
# ---------------------------------------------------------------------------

"""
    seq_lstm_loss(model, ps, st, sequence) -> (loss, st_new)

Negative log-likelihood of `sequence[end, :]` given `sequence[1:end-1, :]`.
"""
function seq_lstm_loss(model, ps, st, sequence::AbstractMatrix{<:Integer})
    T, B = size(sequence)
    T >= 2 || throw(ArgumentError("sequence must have at least 2 time steps"))
    inputs = @view sequence[1:T-1, :]
    targets = @view sequence[T, :]
    logits, st_new = model(inputs, ps, st)
    lp = logsoftmax(logits; dims = 1)
    total = zero(eltype(lp))
    @inbounds for b in 1:B
        total -= lp[Int(targets[b]), b]
    end
    return total / B, st_new
end

"""
    predict_next(model, ps, st, sequence) -> Vector{Int}

Greedy next-event argmax.
"""
function predict_next(model, ps, st, sequence::AbstractMatrix{<:Integer})
    logits, _ = model(sequence, ps, Lux.testmode(st))
    B = size(logits, 2)
    out = Vector{Int}(undef, B)
    @inbounds for b in 1:B
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
