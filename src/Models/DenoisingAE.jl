"""
    DenoisingAE

Denoising / masked autoencoder — Vincent, Larochelle, Bengio &
Manzagol 2008 ("Extracting and Composing Robust Features with
Denoising Autoencoders"), extended in spirit by He et al. 2022
("Masked Autoencoders Are Scalable Vision Learners"). Plan 001
Stage C sibling of KATE / DeepKATE / VQ-VAE / SimCSE.

The contract is deliberately simple:

- [`denoising_ae`] builds a small symmetric Dense encoder / decoder
  `Chain` with a named latent layer. Same (input, batch) → (input,
  batch) shape contract as the thesis autoencoder, so
  [`Anomaly.Instance`] and [`Cluster.Pipeline`] work on it without
  change.
- [`denoising_ae_loss`] corrupts the *input* — either by zeroing a
  fraction of features (MAE-style `mask_rate`) or by adding
  Gaussian noise of scale `σ`, or both — and compares the
  reconstruction to the *clean* target. Returns `(loss, st_new)`
  to fit the rest of the Stage C API.

Why have this alongside KATE / DeepKATE / VQ-VAE / SimCSE? The
four existing embedders all depend on a specific inductive bias
(k-competition, vector quantisation, contrastive dropout). The
denoising AE adds a complementary, bias-light self-supervised
signal — "reconstruct the clean input from a corrupted view" — that
trains cleanly on small corpora and transfers to anomaly scoring
because the reconstruction error surface it learns is, by
construction, robust to the corruption distribution.
"""
module DenoisingAE

using Lux
using Random
using Statistics

export denoising_ae, denoising_ae_loss

# `latent_layer` is NOT exported — `DeepKATE` already exports the same
# name with a different return value, and we don't want two Stage-C
# embedders to clobber each other in `LogClustering`'s namespace.
# Call `DenoisingAE.latent_layer(m)` explicitly if you need the index.

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

"""
    denoising_ae(n::Integer; hidden::Integer = 64, latent::Integer = 16) -> Chain

Symmetric denoising autoencoder for input dimension `n`:

```
Dense(n → hidden, tanh)
Dense(hidden → latent, tanh)       # latent layer
Dense(latent → hidden, tanh)
Dense(hidden → n, σ)
```

The latent layer is position `2` in the `Chain`; query
[`latent_layer`] if you hand the model to
`Anomaly.Instance.latent_distance`.
"""
function denoising_ae(n::Integer;
                      hidden::Integer = 64,
                      latent::Integer = 16)
    return Chain(
        Dense(n => hidden, tanh),
        Dense(hidden => latent, tanh),
        Dense(latent => hidden, tanh),
        Dense(hidden => n, sigmoid),
    )
end

"""
    latent_layer(::Chain) -> Int

The 1-based index of the encoder-output layer for any `Chain`
produced by [`denoising_ae`]. Always `2` for the current topology.
"""
latent_layer(::Chain) = 2

# ---------------------------------------------------------------------------
# Input corruption
# ---------------------------------------------------------------------------
#
# Two knobs, independently applied:
#   - mask_rate ∈ [0, 1): Bernoulli mask that zeros that fraction
#     of input features per sample (MAE-style).
#   - σ ≥ 0: additive Gaussian noise of that scale.
#
# Both default to "off" so callers can turn on only one if they
# want pure masking or pure noise. The corruption is not
# differentiated through — Zygote sees a constant-noise-rate
# tensor even when the mask changes between calls.

function _corrupt(x::AbstractMatrix,
                  mask_rate::Real,
                  σ::Real,
                  rng::AbstractRNG)
    T = eltype(x)
    out = x
    if mask_rate > 0
        keep = T(1) .- T(mask_rate)
        mask = T.(rand(rng, size(x)...) .< keep)
        out = out .* mask
    end
    if σ > 0
        noise = T.(randn(rng, size(x)...)) .* T(σ)
        out = out .+ noise
    end
    return out
end

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

"""
    denoising_ae_loss(model, ps, st, x::AbstractMatrix;
                      mask_rate = 0.3, σ = 0.0,
                      rng = Random.default_rng())
        -> (loss::Real, st_new)

Corrupt `x`, run the model, compare the reconstruction to the
**clean** `x` via mean-squared error. `mask_rate` zeros that
fraction of inputs per sample; `σ` adds Gaussian noise of that
scale. Passing both composes both corruptions.

The corruption is re-drawn every call — training with
`denoising_ae_loss` explores many masks per epoch, the same way
Dropout layers do.
"""
function denoising_ae_loss(model::Chain, ps, st,
                           x::AbstractMatrix{<:Real};
                           mask_rate::Real = 0.3,
                           σ::Real = 0.0,
                           rng::AbstractRNG = Random.default_rng())
    (0.0 <= mask_rate < 1.0) ||
        throw(ArgumentError("mask_rate must be in [0, 1); got $mask_rate"))
    σ >= 0 || throw(ArgumentError("σ must be ≥ 0; got $σ"))
    x̃ = _corrupt(x, mask_rate, σ, rng)
    ŷ, st_new = model(x̃, ps, st)
    loss = mean(abs2, ŷ .- x)
    return loss, st_new
end

end # module DenoisingAE
