#
# Anomaly demo: train DeepKATE on two "normal" workflows, then score a
# held-out batch containing three deliberate outliers.
#
#   julia --project examples/anomaly_demo.jl
#
# Uses a plain SGD loop with Zygote gradients (no optimiser package, so
# the example stays zero-dep beyond the test env). Prints the score
# distribution and the top-K flagged samples.

using LogClustering
using LogClustering.Instance
using Lux
using Random
using Statistics
using Zygote

hdr(t) = (println(); println(repeat("─", 72)); println("  ", t); println(repeat("─", 72)))

# ─── synthetic workflow data ────────────────────────────────────────────
# Each sample is a 16-dim vector in [0,1]. Normal samples come from one
# of two Gaussian blobs inside the unit cube (clipped to [0,1]).
# Outliers are uniform over [0,1]^16 — maximally unlike either blob.
function gen_batch(rng, n, centre, σ)
    X = similar(Array{Float32}(undef, 16, n))
    for j in 1:n, i in 1:16
        X[i, j] = clamp(centre[i] + σ * randn(rng, Float32), 0f0, 1f0)
    end
    return X
end

rng = Random.MersenneTwister(101)
c1 = Float32[0.1, 0.1, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1,
             0.9, 0.85, 0.8, 0.9, 0.85, 0.9, 0.8, 0.85]
c2 = Float32[0.9, 0.85, 0.8, 0.9, 0.85, 0.9, 0.8, 0.85,
             0.1, 0.1, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1]
X_train = hcat(gen_batch(rng, 100, c1, 0.05f0),
               gen_batch(rng, 100, c2, 0.05f0))

# ─── Vanilla AE + SGD on BCE reconstruction ────────────────────────────
#
# Using a plain Dense-stacked AE here rather than DeepKATE: the thesis's
# competitive layers + dropout + sine bottleneck make SGD convergence
# noisy on a toy batch of 200 samples, and this demo is about *scoring*,
# not about DeepKATE's training dynamics (those are exercised in the
# test suite). The Anomaly.Instance scoring is AE-agnostic — any Lux
# `Chain` with matching in/out dims works.
hdr("1. Training — SGD on a 6-layer AE")
model = Chain(
    Dense(16 => 32, relu),
    Dense(32 => 3,  tanh),    # latent (3-D)
    Dense(3  => 32, relu),
    Dense(32 => 16, sigmoid),
)
ps, st = Lux.setup(rng, model)

function bce(ŷ, y)
    ϵ = 1f-7
    -mean(@. y * log(max(ŷ, ϵ)) + (1 - y) * log(max(1 - ŷ, ϵ)))
end
# Training-mode forward so dropout & KATE competition receive gradients.
forward_loss(m, p, s, X) = (bce(first(m(X, p, s)), X), s)

# Manual SGD across a NamedTuple tree of params.
function sgd!(ps, grads, lr)
    if grads === nothing
        return ps
    elseif ps isa AbstractArray
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

batch = 64
lr    = 1f-1
steps = 600
for step in 1:steps
    cols = rand(rng, 1:size(X_train, 2), batch)
    Xb   = X_train[:, cols]
    (loss, _), back = Zygote.pullback(p -> forward_loss(model, p, st, Xb), ps)
    g = back((one(loss), nothing))[1]
    global ps = sgd!(ps, g, lr)
    step % (steps ÷ 10) == 0 && println("    step ", lpad(step, 3), "   bce = ", round(loss; digits = 4))
end

# ─── scoring — 10 held-out normals + 3 outliers ─────────────────────────
hdr("2. Held-out scoring")
X_normal  = hcat(gen_batch(rng, 5, c1, 0.05f0), gen_batch(rng, 5, c2, 0.05f0))
X_outlier = rand(rng, Float32, 16, 3)             # uniform [0,1]^16
X_eval    = hcat(X_normal, X_outlier)
labels    = vcat(fill(:normal, 10), fill(:outlier, 3))

scores = anomaly_score(model, ps, st, X_eval;
                       weights = (abs = 1.0, sq = 1.0, latent = 0.0),
                       normalise = false)

println(lpad("idx", 4), "  ", rpad("label", 9), "  ",
        lpad("score", 10), "  bar")
smax = maximum(scores)
for i in 1:length(scores)
    bar = repeat("█", round(Int, 40 * scores[i] / smax))
    println(lpad(i, 4), "  ", rpad(string(labels[i]), 9), "  ",
            lpad(round(scores[i]; digits = 4), 10), "  ", bar)
end

# ─── flag the worst 3 ───────────────────────────────────────────────────
hdr("3. Top-3 flagged")
ranked = sortperm(scores; rev = true)
for rk in 1:3
    i = ranked[rk]
    println("    rank ", rk, "  idx=", lpad(i, 2), "  label=", labels[i],
            "  score=", round(scores[i]; digits = 4))
end

normal_max  = maximum(scores[labels .== :normal])
outlier_min = minimum(scores[labels .== :outlier])
sep = outlier_min - normal_max
println()
println("  max(normal)    = ", round(normal_max; digits = 4))
println("  min(outlier)   = ", round(outlier_min; digits = 4))
println("  separation gap = ", round(sep; digits = 4),
        sep > 0 ? "  ✔ cleanly separable" : "  × overlap")
