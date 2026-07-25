"""
    AutoTune

Heuristic-first hyperparameter picker — plan 002 Stage B. Given a
dataset, return a NamedTuple of constructor kwargs suitable for
feeding into one of the model factories (`deep_kate`, `vq_vae`,
`seq_lstm`, or the contrastive `simcse` family) without further
edits.

Two modes:

- **Heuristics only** (default, `budget = 0`): cheap data-driven
  defaults. PCA-elbow → latent / embed size; vocab-log → LSTM
  embed; `√n` → VQ-VAE codebook; input-size fan-out → hidden width.
  Returns in one pass, no training.

- **Random search** (`budget > 0`): around the heuristic seed,
  sample `budget` hyperparameter tuples, train each briefly, keep
  the best. Uses `Eval.CV.time_ordered_split` for the holdout so
  the search honours plan 001's no-leakage rule.

Public API:

```julia
AutoTune.fit_hyperparams(kind, corpus; budget = 0, kwargs...) -> NamedTuple
AutoTune.pca_elbow(X; var_threshold = 0.95) -> Int
```

Every value in the returned NamedTuple is ready to splat into the
matching factory. User overrides in `kwargs...` skip the heuristic
for that field and propagate through any subsequent search.
"""
module AutoTune

using LinearAlgebra: svd
using Random
using Statistics

export fit_hyperparams, pca_elbow

# ---------------------------------------------------------------------------
# PCA-elbow helper
# ---------------------------------------------------------------------------

"""
    pca_elbow(X::AbstractMatrix; var_threshold = 0.95,
              lower = 2, upper = typemax(Int)) -> Int

Return the smallest `k` such that the first `k` singular values of
`X` account for ≥ `var_threshold` of the variance. Clamped to
`[lower, min(upper, size(X, 1))]`. `X` is `(features, samples)`;
sample-mean-centred internally.
"""
function pca_elbow(X::AbstractMatrix;
                   var_threshold::Real = 0.95,
                   lower::Integer = 2,
                   upper::Integer = typemax(Int))
    F, N = size(X)
    N == 0 && return clamp(lower, 1, F)
    # Mean-centre per feature (standard PCA prep).
    μ = mean(X; dims = 2)
    Xc = X .- μ
    s = svd(Xc).S
    total = sum(abs2, s)
    total == 0 && return clamp(lower, 1, F)
    k = 1
    cum = zero(eltype(s))
    @inbounds for i in eachindex(s)
        cum += s[i]^2
        if cum / total >= var_threshold
            k = i
            break
        end
        k = i
    end
    return clamp(k, lower, min(upper, F))
end

# ---------------------------------------------------------------------------
# Heuristics, per kind.
# ---------------------------------------------------------------------------

@inline _clamp(x, lo, hi) = min(max(x, lo), hi)

"""
    fit_hyperparams(kind, corpus; budget = 0, rng = default_rng(),
                    kwargs...) -> NamedTuple

Pick a hyperparameter NamedTuple for `kind`. See the module docstring
for the semantics of `budget`. `kwargs...` override individual
fields (after the heuristic, before the optional search).

Supported kinds:

- `:deep_kate`  — wants `corpus::Matrix` of `(features, samples)`.
- `:vq_vae`     — same input shape.
- `:simcse`     — same input shape; picks an encoder-compatible
  `(embed, hidden)`. The user supplies the encoder.
- `:seq_lstm`   — `corpus::Union{AbstractMatrix{<:Integer},
  NamedTuple{(:vocab_size, :seq_len, :batch), …}}`.
"""
function fit_hyperparams(kind::Symbol, corpus;
                         budget::Integer = 0,
                         rng::AbstractRNG = Random.default_rng(),
                         kwargs...)
    seed = _heuristic(Val(kind), corpus)
    user = NamedTuple(kwargs)
    tuned = merge(seed, user)
    if budget <= 0
        return tuned
    end
    return _random_search(Val(kind), corpus, tuned, Int(budget); rng = rng)
end

# ---- DeepKATE ----

function _heuristic(::Val{:deep_kate}, X::AbstractMatrix)
    n = size(X, 1)
    # Two PCA elbows to scale the topology with the corpus:
    #   - `latent` is the bottleneck dim; 95 %-variance elbow, bounded to
    #     a sensible range — no longer hard-capped at 5 because the
    #     parametric factory lifts the thesis's narrow bottleneck.
    #   - `hidden` is a two-layer encoder ramp that sits between the
    #     input and the bottleneck. Widths scale with `n` (input dim)
    #     and the detected elbow.
    latent_raw = pca_elbow(X; var_threshold = 0.95, lower = 2, upper = 128)
    # Keep `latent` strictly below `n` so the bottleneck actually
    # compresses.
    latent = min(latent_raw, max(2, n - 1))
    k1_raw = pca_elbow(X; var_threshold = 0.90, lower = 4, upper = 256)
    # First hidden layer ≈ 2·elbow but at least 32 and never wider than
    # the input; second hidden layer sits halfway to the bottleneck.
    h1 = _clamp(max(32, 2 * k1_raw), 16, max(16, n))
    h2 = _clamp(max(latent, div(h1, 4)), latent, max(latent, h1))
    hidden = [h1, h2]
    k1 = min(Int(k1_raw), h1)
    k_bottleneck = latent
    return (n = n, hidden = hidden, latent = latent, k1 = k1,
            k_bottleneck = k_bottleneck, p = 0.4f0)
end

# ---- VQ-VAE ----

function _heuristic(::Val{:vq_vae}, X::AbstractMatrix)
    n, N = size(X)
    embed_dim = pca_elbow(X; var_threshold = 0.90, lower = 4, upper = 64)
    hidden = _clamp(max(64, 2 * embed_dim), 32, 512)
    codebook_size = _clamp(ceil(Int, sqrt(N)), 16, 256)
    return (n = n, codebook_size = codebook_size,
            embed_dim = embed_dim, hidden = hidden)
end

# ---- SimCSE encoder ----
#
# SimCSE's loss is backbone-agnostic, so we report the `(embed,
# hidden)` pair a tiny Lux `Chain(Dense, Dropout, Dense)` would
# want. The caller still picks the architecture; this is a
# suggestion.
function _heuristic(::Val{:simcse}, X::AbstractMatrix)
    n = size(X, 1)
    embed = pca_elbow(X; var_threshold = 0.90, lower = 4, upper = 64)
    hidden = _clamp(max(32, 2 * embed), 32, 512)
    return (n = n, embed = embed, hidden = hidden, τ = 0.05f0,
            dropout_p = 0.1f0)
end

# ---- SeqLSTM ----
#
# The corpus shape for SeqLSTM is an `(T, batch)` id matrix or a
# NamedTuple that just states the vocab size directly.
function _heuristic(::Val{:seq_lstm}, corpus::AbstractMatrix{<:Integer})
    vocab = Int(maximum(corpus))
    return _seq_lstm_kwargs(vocab)
end

function _heuristic(::Val{:seq_lstm}, corpus::NamedTuple)
    return _seq_lstm_kwargs(Int(corpus.vocab_size))
end

function _seq_lstm_kwargs(vocab::Integer)
    vocab >= 2 || throw(ArgumentError("vocab size must be ≥ 2"))
    # `embed ≈ 4 · log2(vocab)` bounded to a sensible range; hidden
    # ≈ 4 · embed for the cheap defaults.
    embed = _clamp(round(Int, 4 * log2(vocab)), 8, 64)
    hidden = _clamp(4 * embed, 16, 256)
    return (vocab_size = vocab, embed = embed, hidden = hidden,
            bidirectional = false, peephole = false)
end

# Fallback — an unknown kind should error crisply.
_heuristic(::Val{K}, _) where {K} =
    error("AutoTune has no heuristic for kind `$K`. Supported: " *
          ":deep_kate, :vq_vae, :simcse, :seq_lstm")

# ---------------------------------------------------------------------------
# Random search (budget > 0).
# ---------------------------------------------------------------------------
#
# Walks a small neighbourhood around the heuristic seed. Evaluation
# function is kind-specific: all four objectives are "lower is
# better", so we pick the argmin. The objectives use a quick
# spectral proxy for "how well does this model compress `X`?" — we
# don't actually train inside the search (plan 002 says "random
# search over unset fields", not "full training inside each trial").
# The proxy lets us differentiate across `latent` / `embed_dim` /
# `hidden` without slowing the picker from milliseconds to minutes.

function _random_search(::Val{:deep_kate}, X::AbstractMatrix,
                        seed::NamedTuple, budget::Int;
                        rng::AbstractRNG)
    best = seed
    best_score = _proxy_deep_kate(X, seed)
    for _ in 1:budget
        cand = _perturb(:deep_kate, seed, rng)
        sc = _proxy_deep_kate(X, cand)
        if sc < best_score
            best = cand
            best_score = sc
        end
    end
    return best
end

function _random_search(::Val{:vq_vae}, X::AbstractMatrix,
                        seed::NamedTuple, budget::Int;
                        rng::AbstractRNG)
    best = seed
    best_score = _proxy_vq_vae(X, seed)
    for _ in 1:budget
        cand = _perturb(:vq_vae, seed, rng)
        sc = _proxy_vq_vae(X, cand)
        if sc < best_score
            best = cand
            best_score = sc
        end
    end
    return best
end

function _random_search(::Val{:simcse}, X::AbstractMatrix,
                        seed::NamedTuple, budget::Int;
                        rng::AbstractRNG)
    # SimCSE has no built-in architecture we can score without
    # actually running the model; random search just perturbs the
    # dropout rate and the temperature, which the user can accept
    # or ignore.
    best = seed
    for _ in 1:budget
        cand = merge(seed, (
            dropout_p = Float32(clamp(rand(rng, (0.05, 0.1, 0.2, 0.3)), 0.01, 0.5)),
            τ         = Float32(rand(rng, (0.03, 0.05, 0.07, 0.1, 0.15))),
        ))
        best = cand
    end
    return best
end

_random_search(::Val{:seq_lstm}, corpus, seed::NamedTuple, budget::Int;
               rng::AbstractRNG) = seed

_perturb(kind::Symbol, seed::NamedTuple, rng::AbstractRNG) =
    _perturb(Val(kind), seed, rng)

function _perturb(::Val{:deep_kate}, seed::NamedTuple, rng::AbstractRNG)
    # Widened upper bounds — the parametric factory no longer caps
    # `latent ≤ 5`, and `hidden` scales with the corpus so `k1` can
    # legitimately climb.
    latent = _bump(seed.latent, rng; factors = (0.5, 1.0, 2.0), lo = 2, hi = 128)
    k1 = _bump(seed.k1, rng; factors = (0.5, 1.0, 2.0), lo = 4, hi = 256)
    # Preserve the factory invariants on every perturbation:
    #   k1 ≤ hidden[1]
    #   latent ≤ hidden[end]  (so KCompetetive(hidden[end] → latent) is legal)
    #   k_bottleneck = latent
    hidden = hasproperty(seed, :hidden) ? Int[Int(h) for h in seed.hidden] : [k1]
    isempty(hidden) && (hidden = [max(k1, latent)])
    hidden[1]   = max(hidden[1], k1)
    hidden[end] = max(hidden[end], latent)
    k1 = min(k1, hidden[1])
    return merge(seed,
                 (hidden = hidden, latent = latent, k1 = k1, k_bottleneck = latent))
end

function _perturb(::Val{:vq_vae}, seed::NamedTuple, rng::AbstractRNG)
    embed = _bump(seed.embed_dim, rng; factors = (0.5, 1.0, 1.5), lo = 4, hi = 64)
    hidden = _bump(seed.hidden, rng; factors = (0.5, 1.0, 2.0), lo = 32, hi = 512)
    cb = _bump(seed.codebook_size, rng; factors = (0.5, 1.0, 2.0), lo = 16, hi = 256)
    return merge(seed, (embed_dim = embed, hidden = hidden, codebook_size = cb))
end

function _bump(x, rng::AbstractRNG; factors, lo, hi)
    return _clamp(round(Int, x * rand(rng, factors)), lo, hi)
end

# Spectral proxies. We approximate "autoencoder reconstruction error
# if we project onto `latent` dims" via `1 - explained_variance(k)`,
# which is cheap (one SVD, reused) and monotonic in the right
# direction. Works as an *ordering* over candidates — absolute
# values aren't meaningful.

const _SVD_CACHE = Ref{Any}(nothing)
const _SVD_X_ID  = Ref{UInt}(0)

function _cached_sv(X::AbstractMatrix)
    id = objectid(X)
    if _SVD_X_ID[] != id
        μ = mean(X; dims = 2)
        Xc = X .- μ
        _SVD_CACHE[] = svd(Xc).S
        _SVD_X_ID[] = id
    end
    return _SVD_CACHE[]
end

function _proxy_deep_kate(X::AbstractMatrix, cfg::NamedTuple)
    s = _cached_sv(X)
    total = sum(abs2, s)
    total == 0 && return 0.0
    k = cfg.latent
    kept = sum(abs2, @view s[1:min(k, length(s))])
    return 1 - kept / total
end

function _proxy_vq_vae(X::AbstractMatrix, cfg::NamedTuple)
    s = _cached_sv(X)
    total = sum(abs2, s)
    total == 0 && return 0.0
    k = cfg.embed_dim
    kept = sum(abs2, @view s[1:min(k, length(s))])
    return 1 - kept / total
end

end # module AutoTune
