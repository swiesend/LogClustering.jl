"""
    Transformer

State-of-the-art transformer (encoder + decoder) for sequence analysis.
Pure Lux; no external transformer libraries. Both flavours share the
same `TransformerBlock` and differ only in the causal-mask flag and the
training objective.

SOTA component set (LLaMA / Mistral lineage):

- **RMSNorm + Pre-Norm** placement — single trainable scale per channel,
  applied before attention / FFN inside each residual branch.
- **Rotary Position Embeddings (RoPE)** — feature-pair rotations on Q/K
  with a precomputed `(sin, cos)` cache. No learned position table.
- **Grouped-Query Attention (GQA)** — `n_kv_heads ≤ n_heads`; each KV
  head is shared by `n_heads ÷ n_kv_heads` query heads. Reduces to
  vanilla MHA when `n_kv_heads == n_heads`.
- **SwiGLU FFN** — gated MLP with Swish (a.k.a. SiLU) activation; bias-free.
- **Tied LM head** — the embedding's transpose is reused as the output
  projection, halving the language-model parameter count.

I/O contract (matches [`SeqLSTM`]):

- Input: integer matrix `(seqlen, batch)` with 1-based ids.
- Internal hidden tensor: `(d_model, seqlen, batch)`.
- Decoder loss: shift-one cross-entropy over every position.
- Encoder loss: MLM-style cross-entropy over `mask_rate` of positions
  (the reserved `<UNK>` slot at id 1 doubles as `[MASK]`).
- `embed_sequences` returns a `(d_model, batch)` mean-pooled matrix
  ready for the existing kmeans / sparsity clustering pipeline.
- `predict_next` mirrors `SeqLSTM.predict_next` for drop-in classify use.
"""
module Transformer

using Lux
using LuxCore: LuxCore, AbstractLuxLayer, AbstractLuxContainerLayer
using Random
using Statistics
using WeightInitializers: glorot_uniform, ones32
using NNlib: NNlib, softmax, logsoftmax, swish, batched_mul
using ChainRulesCore: @ignore_derivatives

# `predict_next` and `latent_layer` are intentionally *not* exported —
# both names clash with the SeqLSTM / DeepKATE exports already living
# in `LogClustering`. Callers reach for them as `Transformer.predict_next`
# / `Transformer.latent_layer`, which keeps the per-backbone dispatch
# explicit at the call site (the CLI does this).
export RMSNorm, GQAAttention, SwiGLUFFN, TransformerBlock,
       transformer_encoder, transformer_decoder,
       transformer_decoder_loss, transformer_encoder_loss,
       embed_sequences, d_ff_swiglu

# ---------------------------------------------------------------------------
# RMSNorm — Zhang & Sennrich 2019 (used in LLaMA / Mistral / Gemma).
# ---------------------------------------------------------------------------

"""
    RMSNorm(dim::Int; ε = 1f-6)

Root-mean-square layer normalisation. One trainable scale per channel,
no bias, no mean subtraction:

```
y = x ./ sqrt.(mean(x.^2; dims = 1) .+ ε) .* g
```

Operates on the leading feature axis; trailing axes broadcast.
"""
struct RMSNorm <: AbstractLuxLayer
    dim::Int
    ε::Float32
end

RMSNorm(dim::Integer; ε::Real = 1f-6) = RMSNorm(Int(dim), Float32(ε))

LuxCore.initialparameters(rng::AbstractRNG, l::RMSNorm) =
    (g = ones32(rng, l.dim),)

LuxCore.initialstates(::AbstractRNG, ::RMSNorm) = NamedTuple()
LuxCore.parameterlength(l::RMSNorm) = l.dim
LuxCore.statelength(::RMSNorm) = 0

Base.show(io::IO, l::RMSNorm) = print(io, "RMSNorm(", l.dim, ")")

function (l::RMSNorm)(x::AbstractArray, ps, st::NamedTuple)
    rms = sqrt.(mean(x .^ 2; dims = 1) .+ l.ε)
    return (x ./ rms) .* ps.g, st
end

# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE) — Su et al. 2021. "Halves" formulation,
# matching HuggingFace's LLaMA implementation.
# ---------------------------------------------------------------------------

"""
    rope_cache(head_dim, max_seq_len; base = 10_000f0) -> (cos, sin)

Precompute the per-position rotation tables for RoPE. Returns two
`(head_dim, max_seq_len)` matrices; pairs of features (first half /
second half) share the same angle.
"""
function rope_cache(head_dim::Int, max_seq_len::Int; base::Real = 10_000f0)
    iseven(head_dim) ||
        throw(ArgumentError("RoPE needs even head_dim; got $head_dim"))
    half = head_dim ÷ 2
    # θ_i = base^(-2i/d)  for i in 0:half-1
    inv_freq = Float32[Float32(base)^(-2 * (i - 1) / head_dim) for i in 1:half]
    positions = Float32.(0:(max_seq_len - 1))
    angles = positions .* inv_freq'                             # (max_seq_len, half)
    cos_half = cos.(angles)'                                    # (half, max_seq_len)
    sin_half = sin.(angles)'                                    # (half, max_seq_len)
    cos_full = vcat(cos_half, cos_half)                         # (head_dim, max_seq_len)
    sin_full = vcat(sin_half, sin_half)                         # (head_dim, max_seq_len)
    return Float32.(cos_full), Float32.(sin_full)
end

"`rotate_half([x1; x2]) = [-x2; x1]`. Operates on the leading feature axis."
function rotate_half(x::AbstractArray)
    half = size(x, 1) ÷ 2
    x1 = selectdim(x, 1, 1:half)
    x2 = selectdim(x, 1, (half + 1):(2 * half))
    return vcat(-x2, x1)
end

"""
    apply_rope(x, cos_full, sin_full) -> rotated

Rotate `(head_dim, seqlen, ...)` query/key tensor in-place-of-style
(no mutation). `cos_full`, `sin_full` are `(head_dim, ≥seqlen)` and
sliced internally.
"""
function apply_rope(x::AbstractArray, cos_full::AbstractMatrix,
                    sin_full::AbstractMatrix)
    T = size(x, 2)
    cos_t = @view cos_full[:, 1:T]
    sin_t = @view sin_full[:, 1:T]
    # Reshape cos/sin to broadcast over the trailing axes (e.g. heads, batch).
    extra = ntuple(_ -> 1, ndims(x) - 2)
    cos_b = reshape(cos_t, size(cos_t)..., extra...)
    sin_b = reshape(sin_t, size(sin_t)..., extra...)
    return x .* cos_b .+ rotate_half(x) .* sin_b
end

# ---------------------------------------------------------------------------
# Grouped-Query Attention.
# ---------------------------------------------------------------------------

"""
    GQAAttention(d_model; n_heads, n_kv_heads = n_heads,
                 max_seq_len = 1024, causal = false,
                 init_weight = glorot_uniform)

Multi-head self-attention with grouped KV heads. `wq`, `wk`, `wv`, `wo`
are bias-free Dense projections; RoPE is applied to Q and K with a
state-cached `(cos, sin)` table sized for `max_seq_len`.

Input/output shape: `(d_model, seqlen, batch)` with `seqlen ≤ max_seq_len`.
"""
struct GQAAttention{IW} <: AbstractLuxLayer
    d_model::Int
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
    max_seq_len::Int
    causal::Bool
    init_weight::IW
end

function GQAAttention(d_model::Integer;
                      n_heads::Integer,
                      n_kv_heads::Integer = n_heads,
                      max_seq_len::Integer = 1024,
                      causal::Bool = false,
                      init_weight = glorot_uniform)
    d_model % n_heads == 0 ||
        throw(ArgumentError("d_model ($d_model) must be divisible by n_heads ($n_heads)"))
    n_heads % n_kv_heads == 0 ||
        throw(ArgumentError("n_heads ($n_heads) must be divisible by n_kv_heads ($n_kv_heads)"))
    head_dim = Int(d_model) ÷ Int(n_heads)
    iseven(head_dim) ||
        throw(ArgumentError("head_dim ($head_dim) must be even (RoPE constraint)"))
    return GQAAttention(Int(d_model), Int(n_heads), Int(n_kv_heads), head_dim,
                        Int(max_seq_len), causal, init_weight)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GQAAttention)
    H, D = l.n_heads, l.head_dim
    Hkv = l.n_kv_heads
    d_model = l.d_model
    return (
        wq = l.init_weight(rng, H * D,   d_model),
        wk = l.init_weight(rng, Hkv * D, d_model),
        wv = l.init_weight(rng, Hkv * D, d_model),
        wo = l.init_weight(rng, d_model, H * D),
    )
end

function LuxCore.initialstates(::AbstractRNG, l::GQAAttention)
    cos_full, sin_full = rope_cache(l.head_dim, l.max_seq_len)
    return (rope_cos = cos_full, rope_sin = sin_full)
end

LuxCore.parameterlength(l::GQAAttention) =
    l.d_model * (l.n_heads * l.head_dim) +
    2 * l.d_model * (l.n_kv_heads * l.head_dim) +
    (l.n_heads * l.head_dim) * l.d_model

LuxCore.statelength(l::GQAAttention) = 2 * l.head_dim * l.max_seq_len

Base.show(io::IO, l::GQAAttention) =
    print(io, "GQAAttention(", l.d_model, ", n_heads=", l.n_heads,
          ", n_kv_heads=", l.n_kv_heads,
          l.causal ? ", causal" : "", ")")

# Repeat KV heads so they line up with the query heads. For GQA each
# kv-head is shared by `group` query heads; broadcasting + reshape avoids
# `repeat` overhead and keeps the gradient path simple.
function _repeat_kv(x::AbstractArray, group::Int)
    group == 1 && return x
    head_dim, seqlen, n_kv, batch = size(x)
    # Insert a singleton "group" axis after the kv-heads axis, then
    # broadcast to (head_dim, seqlen, n_kv, group, batch). The trailing
    # `ones` reshape lets the broadcast allocate a contiguous block.
    y = reshape(x, head_dim, seqlen, n_kv, 1, batch) .+
        reshape(zeros(eltype(x), 1, 1, 1, group, 1),
                1, 1, 1, group, 1)
    return reshape(y, head_dim, seqlen, n_kv * group, batch)
end

# Strict upper-triangular causal mask added to attention scores
# `(seqlen, seqlen, ...)`. Marked `@ignore_derivatives` so Zygote
# doesn't try to differentiate the constant.
function _causal_mask(::Type{T}, seqlen::Int) where {T}
    @ignore_derivatives begin
        m = fill(zero(T), seqlen, seqlen)
        @inbounds for j in 1:seqlen, i in 1:seqlen
            if j > i
                m[i, j] = T(-Inf)
            end
        end
        return m
    end
end

function (l::GQAAttention)(x::AbstractArray{T, 3}, ps, st::NamedTuple) where {T}
    d_model, seqlen, batch = size(x)
    d_model == l.d_model ||
        throw(DimensionMismatch("expected d_model=$(l.d_model), got $d_model"))
    seqlen <= l.max_seq_len ||
        throw(ArgumentError("seqlen $seqlen exceeds max_seq_len $(l.max_seq_len)"))

    H, Hkv, D = l.n_heads, l.n_kv_heads, l.head_dim
    group = H ÷ Hkv

    # Project Q, K, V via 2-D matmul (flatten time × batch).
    x_2d = reshape(x, d_model, seqlen * batch)
    q = ps.wq * x_2d                                            # (H·D, seqlen·batch)
    k = ps.wk * x_2d                                            # (Hkv·D, seqlen·batch)
    v = ps.wv * x_2d

    # Reshape to (head_dim, n_heads, seqlen, batch) then permute so
    # `(head_dim, seqlen, n_heads, batch)` — RoPE wants (head_dim, seqlen, ...).
    q4 = permutedims(reshape(q, D, H,   seqlen, batch), (1, 3, 2, 4))
    k4 = permutedims(reshape(k, D, Hkv, seqlen, batch), (1, 3, 2, 4))
    v4 = permutedims(reshape(v, D, Hkv, seqlen, batch), (1, 3, 2, 4))

    # Apply RoPE to Q and K (V is not rotated).
    cos_full = st.rope_cos
    sin_full = st.rope_sin
    q4 = apply_rope(q4, cos_full, sin_full)
    k4 = apply_rope(k4, cos_full, sin_full)

    # GQA: broadcast each kv-head to `group` query heads.
    k4 = _repeat_kv(k4, group)
    v4 = _repeat_kv(v4, group)

    # Merge (n_heads, batch) into one batch axis for `batched_mul`.
    q3 = reshape(q4, D, seqlen, H * batch)
    k3 = reshape(k4, D, seqlen, H * batch)
    v3 = reshape(v4, D, seqlen, H * batch)

    # Scaled dot-product: scores[i, j, hb] = (q[:, i, hb])' * k[:, j, hb]
    # i.e. `batched_mul(batched_transpose(Q), K)`.
    scale = T(1) / T(sqrt(D))
    qT = permutedims(q3, (2, 1, 3))                             # (seqlen, D, H·batch)
    scores = batched_mul(qT, k3) .* scale                       # (seqlen, seqlen, H·batch)

    if l.causal
        mask = T.(_causal_mask(T, seqlen))
        scores = scores .+ reshape(mask, seqlen, seqlen, 1)
    end

    attn = softmax(scores; dims = 2)                            # over key axis

    # out[d, t_q, hb] = sum_{t_k} V[d, t_k, hb] * attn[t_q, t_k, hb]
    attnT = permutedims(attn, (2, 1, 3))                        # (seqlen_k, seqlen_q, H·batch)
    out3 = batched_mul(v3, attnT)                               # (D, seqlen, H·batch)

    # Reshape back to (head_dim, seqlen, n_heads, batch) → permute →
    # (n_heads·head_dim, seqlen, batch) → 2-D for the output projection.
    out4 = reshape(out3, D, seqlen, H, batch)
    out4p = permutedims(out4, (1, 3, 2, 4))                     # (D, H, seqlen, batch)
    out_concat = reshape(out4p, H * D, seqlen * batch)
    y_2d = ps.wo * out_concat
    y = reshape(y_2d, d_model, seqlen, batch)
    return y, st
end

# ---------------------------------------------------------------------------
# SwiGLU FFN — Shazeer 2020. Gated MLP with SiLU/Swish activation.
# `d_ff = round(8/3 · d_model · ffn_mult / 64) · 64` (LLaMA convention).
# ---------------------------------------------------------------------------

"Default SwiGLU hidden size: rounds `8/3 · d_model · ffn_mult` to a multiple of 64."
function d_ff_swiglu(d_model::Integer; ffn_mult::Real = 4)
    raw = (8 / 3) * d_model * ffn_mult / 64
    return max(64, 64 * max(1, round(Int, raw)))
end

"""
    SwiGLUFFN(d_model; d_ff = d_ff_swiglu(d_model),
              init_weight = glorot_uniform)

Gated FFN: `silu(W1 x) ⊙ (W3 x)` projected back via `W2`. Bias-free.
"""
struct SwiGLUFFN{IW} <: AbstractLuxLayer
    d_model::Int
    d_ff::Int
    init_weight::IW
end

function SwiGLUFFN(d_model::Integer;
                   d_ff::Integer = d_ff_swiglu(d_model),
                   init_weight = glorot_uniform)
    return SwiGLUFFN(Int(d_model), Int(d_ff), init_weight)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SwiGLUFFN)
    return (
        w1 = l.init_weight(rng, l.d_ff,    l.d_model),
        w3 = l.init_weight(rng, l.d_ff,    l.d_model),
        w2 = l.init_weight(rng, l.d_model, l.d_ff),
    )
end

LuxCore.initialstates(::AbstractRNG, ::SwiGLUFFN) = NamedTuple()
LuxCore.parameterlength(l::SwiGLUFFN) = 3 * l.d_model * l.d_ff
LuxCore.statelength(::SwiGLUFFN) = 0

Base.show(io::IO, l::SwiGLUFFN) =
    print(io, "SwiGLUFFN(", l.d_model, " => ", l.d_ff, ")")

function (l::SwiGLUFFN)(x::AbstractArray, ps, st::NamedTuple)
    d_model = size(x, 1)
    trailing = size(x)[2:end]
    x_2d = reshape(x, d_model, prod(trailing))
    gate = swish.(ps.w1 * x_2d)
    up   = ps.w3 * x_2d
    h    = gate .* up
    y_2d = ps.w2 * h
    y    = reshape(y_2d, d_model, trailing...)
    return y, st
end

# ---------------------------------------------------------------------------
# TransformerBlock — Pre-Norm residual: SkipConnection wrapping
# (RMSNorm → attention → dropout) and (RMSNorm → SwiGLU → dropout).
# ---------------------------------------------------------------------------

"""
    TransformerBlock(d_model; n_heads, n_kv_heads = n_heads,
                     d_ff = d_ff_swiglu(d_model),
                     max_seq_len = 1024, causal = false, dropout = 0.0)

Pre-Norm residual block:

```
x ← x + Dropout(GQA(RMSNorm(x)))
x ← x + Dropout(SwiGLU(RMSNorm(x)))
```

Returns a Lux `Chain` of two `SkipConnection`s — fully integrates with
the existing `Lux.setup` / `(ps, st)` plumbing.
"""
function TransformerBlock(d_model::Integer;
                          n_heads::Integer,
                          n_kv_heads::Integer = n_heads,
                          d_ff::Integer = d_ff_swiglu(d_model),
                          max_seq_len::Integer = 1024,
                          causal::Bool = false,
                          dropout::Real = 0.0)
    attn = GQAAttention(d_model;
                        n_heads = n_heads,
                        n_kv_heads = n_kv_heads,
                        max_seq_len = max_seq_len,
                        causal = causal)
    return Chain(
        SkipConnection(
            Chain(RMSNorm(d_model),
                  attn,
                  Dropout(Float32(dropout))),
            +),
        SkipConnection(
            Chain(RMSNorm(d_model),
                  SwiGLUFFN(d_model; d_ff = d_ff),
                  Dropout(Float32(dropout))),
            +),
    )
end

# ---------------------------------------------------------------------------
# Encoder & decoder factories.
# ---------------------------------------------------------------------------

function _stack_blocks(n_layers::Integer, builder)
    return Tuple(builder(i) for i in 1:Int(n_layers))
end

"""
    transformer_encoder(vocab_size; d_model = 128, n_layers = 4,
                        n_heads = 8, n_kv_heads = n_heads,
                        ffn_mult = 4, max_seq_len = 1024, dropout = 0.0)
        -> Chain

Bidirectional transformer encoder. Output of the chain is the
`(d_model, seqlen, batch)` post-norm hidden state; use
[`embed_sequences`] for a `(d_model, batch)` mean-pooled embedding or
[`transformer_encoder_loss`] for MLM training.
"""
function transformer_encoder(vocab_size::Integer;
                             d_model::Integer = 128,
                             n_layers::Integer = 4,
                             n_heads::Integer = 8,
                             n_kv_heads::Integer = n_heads,
                             ffn_mult::Real = 4,
                             max_seq_len::Integer = 1024,
                             dropout::Real = 0.0)
    d_ff = d_ff_swiglu(d_model; ffn_mult = ffn_mult)
    blocks = _stack_blocks(n_layers, _ ->
        TransformerBlock(d_model;
                         n_heads = n_heads,
                         n_kv_heads = n_kv_heads,
                         d_ff = d_ff,
                         max_seq_len = max_seq_len,
                         causal = false,
                         dropout = dropout))
    return Chain(
        Embedding(vocab_size => d_model),
        blocks...,
        RMSNorm(d_model),
    )
end

"""
    transformer_decoder(vocab_size; d_model = 128, n_layers = 4,
                        n_heads = 8, n_kv_heads = n_heads,
                        ffn_mult = 4, max_seq_len = 1024, dropout = 0.0)
        -> Chain

Causal decoder transformer. Output is the `(d_model, seqlen, batch)`
post-norm hidden state; the LM head is tied to the embedding's weight
inside [`transformer_decoder_loss`] / [`predict_next`].
"""
function transformer_decoder(vocab_size::Integer;
                             d_model::Integer = 128,
                             n_layers::Integer = 4,
                             n_heads::Integer = 8,
                             n_kv_heads::Integer = n_heads,
                             ffn_mult::Real = 4,
                             max_seq_len::Integer = 1024,
                             dropout::Real = 0.0)
    d_ff = d_ff_swiglu(d_model; ffn_mult = ffn_mult)
    blocks = _stack_blocks(n_layers, _ ->
        TransformerBlock(d_model;
                         n_heads = n_heads,
                         n_kv_heads = n_kv_heads,
                         d_ff = d_ff,
                         max_seq_len = max_seq_len,
                         causal = true,
                         dropout = dropout))
    return Chain(
        Embedding(vocab_size => d_model),
        blocks...,
        RMSNorm(d_model),
    )
end

"""
    latent_layer(::Chain) -> Int

Position of the final RMSNorm in either encoder or decoder Chains.
Mirrors `DeepKATE.latent_layer` so the shared CLI clustering path can
reach the post-norm hidden state with `_forward_through(model, ps, st,
X, latent_layer(model))`.
"""
latent_layer(model::Chain) = length(model.layers)

# ---------------------------------------------------------------------------
# Loss functions.
# ---------------------------------------------------------------------------

# Tied LM head: project hidden states back into vocab space using the
# embedding's transpose.
function _tied_logits(h::AbstractArray{T, 3}, embed_w::AbstractMatrix{T}) where {T}
    d_model, seqlen, batch = size(h)
    h_2d = reshape(h, d_model, seqlen * batch)
    logits_2d = embed_w' * h_2d                                 # (vocab, seqlen·batch)
    vocab = size(embed_w, 2)
    return reshape(logits_2d, vocab, seqlen, batch)
end

"""
    transformer_decoder_loss(model, ps, st, sequence) -> (loss, st_new)

Modern next-token training: shift-one cross-entropy over every position.
Given `sequence::Matrix{Int}` of shape `(seqlen, batch)`, the model
sees `sequence[1:end-1, :]` and is scored on `sequence[2:end, :]`.
The output projection is tied to `ps.layer_1.weight` (the embedding).
"""
function transformer_decoder_loss(model, ps, st,
                                  sequence::AbstractMatrix{<:Integer})
    T_, B = size(sequence)
    T_ >= 2 || throw(ArgumentError("sequence must have at least 2 time steps"))
    inputs  = @view sequence[1:(T_ - 1), :]
    targets = @view sequence[2:T_, :]
    h, st_new = model(inputs, ps, st)                           # (d_model, T-1, B)
    logits = _tied_logits(h, ps.layer_1.weight)
    lp = logsoftmax(logits; dims = 1)
    Tm = T_ - 1
    total = zero(eltype(lp))
    @inbounds for b in 1:B, t in 1:Tm
        total -= lp[Int(targets[t, b]), t, b]
    end
    return total / (Tm * B), st_new
end

"""
    transformer_encoder_loss(model, ps, st, sequence;
                             mask_rate = 0.15, mask_id = 1, rng)
        -> (loss, st_new)

Masked-language-model objective. A fraction `mask_rate` of the input
positions are replaced with `mask_id` (defaults to the reserved `<UNK>`
slot at id 1, which the existing `Featurise.Vocabulary` reserves) and
the model is scored only on those masked positions. Uses the tied LM
head built from `ps.layer_1.weight`.
"""
function transformer_encoder_loss(model, ps, st,
                                  sequence::AbstractMatrix{<:Integer};
                                  mask_rate::Real = 0.15,
                                  mask_id::Integer = 1,
                                  rng::AbstractRNG = Random.default_rng())
    T_, B = size(sequence)
    T_ >= 1 || throw(ArgumentError("sequence must have at least 1 step"))
    0 < mask_rate < 1 ||
        throw(ArgumentError("mask_rate must be in (0,1); got $mask_rate"))
    # Choose at least one position per column so the loss has signal.
    mask = @ignore_derivatives begin
        m = rand(rng, Float32, T_, B) .< Float32(mask_rate)
        @inbounds for b in 1:B
            if !any(@view m[:, b])
                m[rand(rng, 1:T_), b] = true
            end
        end
        m
    end
    masked_input = @ignore_derivatives ifelse.(mask, Int(mask_id), Int.(sequence))
    h, st_new = model(masked_input, ps, st)
    logits = _tied_logits(h, ps.layer_1.weight)
    lp = logsoftmax(logits; dims = 1)
    total = zero(eltype(lp))
    n_masked = 0
    @inbounds for b in 1:B, t in 1:T_
        if mask[t, b]
            total -= lp[Int(sequence[t, b]), t, b]
            n_masked += 1
        end
    end
    return total / max(1, n_masked), st_new
end

# ---------------------------------------------------------------------------
# Inference helpers.
# ---------------------------------------------------------------------------

"""
    embed_sequences(model, ps, st, sequence; pool = :mean) -> Matrix

Run the transformer in test mode and pool the final hidden states into
a `(d_model, batch)` matrix consumable by the existing kmeans / sparsity
clustering path. `pool ∈ (:mean, :last)`.
"""
function embed_sequences(model, ps, st,
                         sequence::AbstractMatrix{<:Integer};
                         pool::Symbol = :mean)
    h, _ = model(sequence, ps, Lux.testmode(st))                # (d_model, T, B)
    if pool === :mean
        return dropdims(mean(h; dims = 2); dims = 2)
    elseif pool === :last
        return h[:, end, :]
    else
        throw(ArgumentError("pool must be :mean or :last; got $pool"))
    end
end

"""
    predict_next(model, ps, st, sequence) -> Vector{Int}

Greedy next-token argmax from the decoder's last position. Mirrors
`SeqLSTM.predict_next` so `cmd_classify` can dispatch to either
backbone with the same signature.
"""
function predict_next(model, ps, st,
                      sequence::AbstractMatrix{<:Integer})
    h, _ = model(sequence, ps, Lux.testmode(st))                # (d_model, T, B)
    h_last = h[:, end, :]                                       # (d_model, B)
    embed_w = ps.layer_1.weight                                 # (d_model, vocab)
    logits = embed_w' * h_last                                  # (vocab, B)
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

end # module Transformer
