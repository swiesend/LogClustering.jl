"""
    SimCSE

Contrastive sentence-encoder loss — Gao, Yao & Chen 2021
([arXiv 2104.08821](https://arxiv.org/abs/2104.08821)). Plan 001
Stage C sibling of KATE / DeepKATE / VQ-VAE. The idea is to train any
Lux encoder with InfoNCE: each input is fed through the encoder
*twice* with different Dropout masks, yielding a positive pair
`(h_i, h_i⁺)`; everyone else in the batch is a negative. Pulls
syntactically-similar log lines together in the latent space and
pushes dissimilar ones apart without any label supervision.

API surface — backbone-agnostic, deliberately thin:

- [`simcse_loss(h, h⁺; τ)`] — the loss from two already-computed
  `(embed, batch)` representations.
- [`simcse_step_loss(model, ps, st, x; τ)`] — convenience that runs
  the model twice and applies the loss. Requires at least one
  Dropout layer in `model` (else the two passes are identical and
  the loss is `log(batch)` — the uniform baseline).

The implementation deliberately does *not* prescribe an encoder
backbone; plan 001 Stage C calls out SimCSE as a training recipe
you apply on top of any sentence encoder (BGE, KATE, a tiny Dense
stack, …).
"""
module SimCSE

using Lux
using LinearAlgebra: normalize
using Statistics
using NNlib: logsoftmax

export simcse_loss, simcse_step_loss

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

"""
    simcse_loss(h::AbstractMatrix, hp::AbstractMatrix; τ = 0.05) -> Float

InfoNCE loss over `(embed, batch)` columns. Expects `size(h) ==
size(hp)`. Returns the mean negative log-likelihood of the
positive pair at column `i` being ranked first among all `hp`
columns by cosine similarity scaled by `1/τ`.
"""
function simcse_loss(h::AbstractMatrix, hp::AbstractMatrix; τ::Real = 0.05)
    size(h) == size(hp) ||
        throw(DimensionMismatch("h $(size(h)) and h⁺ $(size(hp)) must match"))
    size(h, 2) >= 2 ||
        throw(ArgumentError("batch size must be ≥ 2; got $(size(h, 2))"))
    # L2-normalise columns — cosine similarity becomes a dot product.
    h_n  = _l2_columns(h)
    hp_n = _l2_columns(hp)
    # sim[i, j] = cos(h[:, i], hp[:, j])
    sim = (h_n' * hp_n) ./ eltype(h)(τ)
    # Cross-entropy with targets {1, 2, …, B}.
    lp = logsoftmax(sim; dims = 2)
    B = size(sim, 1)
    total = zero(eltype(lp))
    @inbounds for i in 1:B
        total -= lp[i, i]
    end
    return total / B
end

function _l2_columns(X::AbstractMatrix)
    norms = sqrt.(sum(abs2, X; dims = 1))
    safe = max.(norms, eps(eltype(X)))
    return X ./ safe
end

"""
    simcse_step_loss(model, ps, st, x; τ = 0.05) -> (loss, st_final)

Run `model` twice on the same input `x`. Each forward pass draws a
fresh Dropout mask (Lux's Dropout consumes/advances the RNG stored
in `st`), so the two outputs form a `(h_i, h_i⁺)` positive pair.
Returns the SimCSE loss and the state after both passes. Input
`x` is any shape the encoder accepts; the encoder's output must be
`(embed, batch)`.
"""
function simcse_step_loss(model, ps, st, x; τ::Real = 0.05)
    h,  st1 = model(x, ps, st)
    hp, st2 = model(x, ps, st1)
    return simcse_loss(h, hp; τ = τ), st2
end

end # module SimCSE