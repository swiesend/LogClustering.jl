"""
    VQVAE

Vector-Quantized Variational Autoencoder — van den Oord, Vinyals &
Kavukcuoglu 2017 ([arXiv 1711.00937](https://arxiv.org/abs/1711.00937)).
Plan 001 Stage C sibling of KATE / DeepKATE; the codebook *is* the
cluster vocabulary, so the assigned code per input becomes its
cluster id "for free".

Three pieces:

- [`VectorQuantizer`] — `AbstractLuxLayer` that snaps an encoder
  output `z_e` to the nearest codebook vector `z_q`, with a
  straight-through gradient so training pretends the VQ step is the
  identity.
- [`vq_vae`] — `Chain` constructor: Dense encoder → `VectorQuantizer`
  → Dense decoder → sigmoid reconstruction.
- [`vq_vae_loss`] — reconstruction + codebook + commitment (loss
  components 1/2/3 of equation 3 in the paper).
- [`assign_codes`] — run the encoder + VQ in test mode and return
  the `(batch,)`-length `Vector{Int}` of assigned code indices.
"""
module VQVAE

using Lux
using LuxCore: LuxCore, AbstractLuxLayer
using Random: AbstractRNG
using ChainRulesCore: ChainRulesCore, NoTangent, @ignore_derivatives
using Statistics
using WeightInitializers: glorot_uniform

export VectorQuantizer, vq_vae, vq_vae_loss, assign_codes

# ---------------------------------------------------------------------------
# Vector quantizer
# ---------------------------------------------------------------------------

"""
    VectorQuantizer(embed_dim, codebook_size; init_weight = glorot_uniform)

The quantisation step of van den Oord 2017. Parameters: a single
codebook matrix of shape `(embed_dim, codebook_size)`. Forward input
is `(embed_dim, batch)`; output is the quantised `(embed_dim, batch)`
snap-to-nearest-code result, with a straight-through identity
gradient so the upstream encoder can still be trained by
reconstruction.

The layer writes the latest assignment into its state under key
`:codes` (a `Vector{Int}`) and the raw encoder output under
`:z_e` for the loss function to read without re-running the
encoder.
"""
struct VectorQuantizer{IW} <: AbstractLuxLayer
    embed_dim::Int
    codebook_size::Int
    init_weight::IW
end

function VectorQuantizer(embed_dim::Integer, codebook_size::Integer;
                         init_weight = glorot_uniform)
    embed_dim >= 1 || throw(ArgumentError("embed_dim must be ≥ 1"))
    codebook_size >= 1 || throw(ArgumentError("codebook_size must be ≥ 1"))
    return VectorQuantizer(Int(embed_dim), Int(codebook_size), init_weight)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::VectorQuantizer)
    return (codebook = l.init_weight(rng, l.embed_dim, l.codebook_size),)
end

function LuxCore.initialstates(::AbstractRNG, l::VectorQuantizer)
    return (codes = Int[], z_e = zeros(Float32, l.embed_dim, 0))
end

LuxCore.parameterlength(l::VectorQuantizer) = l.embed_dim * l.codebook_size
LuxCore.statelength(::VectorQuantizer) = 0

function Base.show(io::IO, l::VectorQuantizer)
    print(io, "VectorQuantizer(", l.embed_dim, " => ",
          l.codebook_size, ")")
end

function (l::VectorQuantizer)(z_e::AbstractMatrix{T}, ps, st::NamedTuple) where {T}
    size(z_e, 1) == l.embed_dim ||
        throw(DimensionMismatch("expected $(l.embed_dim)-dim input, got $(size(z_e, 1))"))
    codebook = ps.codebook                       # (embed_dim, K)
    # Squared distances: ‖z - e_k‖² = ‖z‖² + ‖e_k‖² − 2·⟨z, e_k⟩.
    z_sq = @ignore_derivatives sum(abs2, z_e; dims = 1)          # (1, batch)
    e_sq = @ignore_derivatives sum(abs2, codebook; dims = 1)     # (1, K)
    inner = @ignore_derivatives codebook' * z_e                  # (K, batch)
    dists = @ignore_derivatives e_sq' .+ z_sq .- T(2) .* inner   # (K, batch)
    codes = @ignore_derivatives _argmin_per_column(dists)        # (batch,)
    z_q = @ignore_derivatives codebook[:, codes]                 # (embed_dim, batch)
    # Straight-through estimator: forward returns z_q, gradient
    # flows as if the layer were the identity.
    y = z_e .+ _stop_grad(z_q .- z_e)
    st_new = merge(st, (codes = codes, z_e = collect(z_e)))
    return y, st_new
end

function _argmin_per_column(D::AbstractMatrix)
    n = size(D, 2)
    out = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        best_i = 1
        best_v = D[1, j]
        for i in 2:size(D, 1)
            v = D[i, j]
            if v < best_v
                best_v = v
                best_i = i
            end
        end
        out[j] = best_i
    end
    return out
end

# Identity under Zygote / ChainRules; gradient is NoTangent so no
# flow through the wrapped value. Used for the straight-through
# estimator's non-differentiable shift.
_stop_grad(x) = x
ChainRulesCore.rrule(::typeof(_stop_grad), x) = (x, _ -> (NoTangent(), NoTangent()))

# ---------------------------------------------------------------------------
# VQ-VAE model + loss
# ---------------------------------------------------------------------------

"""
    vq_vae(n; codebook_size = 64, embed_dim = 16, hidden = 64) -> Chain

10-layer-ish reference VQ-VAE for `n`-dim input:

    Dense(n → hidden, relu)
    Dense(hidden → embed_dim)
    VectorQuantizer(embed_dim, codebook_size)
    Dense(embed_dim → hidden, relu)
    Dense(hidden → n, sigmoid)

The first two Denses are the encoder, the `VectorQuantizer` does the
discrete snap, the last two are the decoder. `assign_codes` reaches
the layer-3 state to recover the per-sample code id.
"""
function vq_vae(n::Integer;
                codebook_size::Integer = 64,
                embed_dim::Integer = 16,
                hidden::Integer = 64)
    return Chain(
        Dense(n => hidden, relu),
        Dense(hidden => embed_dim),
        VectorQuantizer(embed_dim, codebook_size),
        Dense(embed_dim => hidden, relu),
        Dense(hidden => n, sigmoid),
    )
end

"""
    vq_vae_loss(model, ps, st, x; β = 0.25) -> (loss, st_new)

Van den Oord equation 3:

```
L = Σᵢ |xᵢ − x̂ᵢ| + ‖sg(z_e) − z_q‖²   +   β · ‖z_e − sg(z_q)‖²
    ├─────────┤   ├────────────────┤       ├───────────────────┤
     recon        codebook loss             commitment loss
```

The codebook loss pulls the picked code toward `z_e`; the commitment
loss penalises an encoder that wanders away from the codebook.
`β = 0.25` is the paper's default.
"""
function vq_vae_loss(model::Chain, ps, st,
                     x::AbstractArray{<:Real}; β::Real = 0.25)
    y, st_new = model(x, ps, st)
    # Reconstruction: L1 matches sigmoid-output AE convention; BCE is
    # available to callers who want it.
    recon = mean(abs.(x .- y))
    z_e = st_new.layer_3.z_e
    codes = st_new.layer_3.codes
    # z_q = codebook[:, codes]; recompute from ps so gradients land
    # on the codebook matrix, not on the cached matrix in st.
    codebook = ps.layer_3.codebook
    z_q = codebook[:, codes]
    codebook_loss = mean(abs2, z_q .- _stop_grad(z_e))
    commitment_loss = mean(abs2, z_e .- _stop_grad(z_q))
    loss = recon + codebook_loss + Float64(β) * commitment_loss
    return loss, st_new
end

"""
    assign_codes(model, ps, st, X) -> Vector{Int}

Run the encoder + VQ in test mode and return the 1-based code id
assigned to every column of `X`. This is the VQ-VAE equivalent of
`Sparsity.sparsity_clusters` for KATE.
"""
function assign_codes(model::Chain, ps, st, X::AbstractMatrix)
    _, st_new = model(X, ps, Lux.testmode(st))
    return copy(st_new.layer_3.codes)
end

end # module VQVAE