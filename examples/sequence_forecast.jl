#
# Sequence-forecast demo: train `SeqLSTM` to recognise a deterministic
# workflow, then predict the next event of several held-out prefixes.
#
#   julia --project examples/sequence_forecast.jl
#
# Workflow: the pattern 1 → 2 → 3 → 4 → 5 → 1 → 2 → … is broken up into
# overlapping windows of length 6. The model sees windows 1..5 and must
# predict the 6th event. On convergence the predictions should be
# deterministic and the per-class confidence should be concentrated.

using LogClustering
using LogClustering.SeqLSTM
using Lux
using Random
using Statistics
using NNlib: softmax
using Zygote

hdr(t) = (println(); println(repeat("─", 64)); println("  ", t); println(repeat("─", 64)))

# ─── build a corpus of length-6 windows over a repeating pattern ────────
const VOCAB = 5                                 # events {1,…,5}
const PATTERN = [1, 2, 3, 4, 5]                 # deterministic cycle
function make_windows(; n = 120, seqlen = 6, offset_jitter = true, rng)
    seqs = Array{Int}(undef, seqlen, n)
    for j in 1:n
        off = offset_jitter ? rand(rng, 0:VOCAB-1) : 0
        for t in 1:seqlen
            seqs[t, j] = PATTERN[((t - 1 + off) % VOCAB) + 1]
        end
    end
    return seqs
end

rng = Random.MersenneTwister(7)
train = make_windows(; n = 200, rng = rng)

# ─── model + SGD ────────────────────────────────────────────────────────
hdr("1. Training a 1-layer LSTM (embed=8, hidden=24) for 300 steps")
model = seq_lstm(VOCAB; embed = 8, hidden = 24)
ps, st = Lux.setup(rng, model)

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

batch = 32
lr    = 1f-1
steps = 300
history = Float64[]
for step in 1:steps
    cols = rand(rng, 1:size(train, 2), batch)
    Xb = train[:, cols]
    (loss, _), back = Zygote.pullback(p -> seq_lstm_loss(model, p, st, Xb), ps)
    g = back((one(loss), nothing))[1]
    global ps = sgd!(ps, g, lr)
    step % 30 == 0 && (
        push!(history, loss);
        println("    step ", lpad(step, 3), "   nll = ", round(loss; digits = 4))
    )
end

# ─── predict on held-out prefixes, report top-1 and confidence ──────────
hdr("2. Held-out predictions (5-event history, forecast the 6th)")
test = make_windows(; n = 12, rng = Random.MersenneTwister(99))
hist    = test[1:5, :]
target  = test[6, :]
pred    = predict_next(model, ps, st, hist)
logits, _ = model(hist, ps, Lux.testmode(st))
probs   = softmax(logits; dims = 1)

println(lpad("idx", 4), "  ", "history           →  expect | predict | p(predict) | correct")
for j in 1:size(hist, 2)
    p_pred = probs[pred[j], j]
    correct = pred[j] == target[j] ? "✔" : "×"
    println(lpad(j, 4), "  ",
            rpad(string(hist[:, j]), 18), " →  ",
            lpad(target[j], 6), "   ",
            lpad(pred[j], 7), "    ",
            lpad(round(p_pred; digits = 3), 7), "     ", correct)
end

acc = mean(pred .== target)
println()
println("  accuracy on held-out batch: ", round(100acc; digits = 1), "%")
println("  final training nll       : ",
        isempty(history) ? "-" : string(round(history[end]; digits = 4)),
        "  (uniform baseline = log(", VOCAB, ") = ",
        round(log(VOCAB); digits = 4), ")")
