"""
    SoftKATE

Modernised descendant of DeepKATE. Keeps the competitive-bottleneck +
reconstruction idea but replaces the three pain points measured in
this session:

1. **Hard top-k k-winners** (`KATE.KCompetetive`) → **Gumbel-softmax
   soft competition** ([`GumbelSoftCompetetive`]). Gradients flow
   through the gate instead of being stopped by `@ignore_derivatives`;
   annealing `temperature: 1.0 → 0.1` during training lets the
   representation start soft and tighten toward sparsity.
2. **BCE reconstruction only** → **joint BCE + SimCSE** ([`soft_kate_loss`]).
   Reconstruction trains the AE; a contrastive term on the bottleneck
   directly shapes cluster geometry (the thing pure AE training does
   not touch, as measured on Thunderbird).
3. **Thesis activations (`tanh`, `σ` everywhere, `sin` bottleneck)**
   → **`gelu` intermediates + LayerNorm + `tanh` bottleneck**. Smooth
   gradients; stable training across widths.

The factory stays parametric like the modern DeepKATE — `hidden`
vector + `latent` — so callers can scale to the corpus. The bottleneck
is the `GumbelSoftCompetetive` layer; `latent_layer(model)` reports
its index so downstream clustering / anomaly code reuses unchanged.
"""
module SoftKATE

using Lux
using LuxCore: LuxCore, AbstractLuxLayer
using Random: AbstractRNG, default_rng
using Statistics: mean
using NNlib: softmax, logsoftmax, gelu, sigmoid
using WeightInitializers: glorot_uniform, zeros32
using ChainRulesCore: @ignore_derivatives
using ..SimCSE: simcse_loss

export GumbelSoftCompetetive, soft_kate, soft_kate_loss, latent_layer,
       set_temperature, anneal_temperature

# ---------------------------------------------------------------------------
# Gumbel-softmax competitive layer
# ---------------------------------------------------------------------------

"""
    GumbelSoftCompetetive(in => out, σ = gelu;
                          init_temperature = 1.0,
                          init_weight = glorot_uniform,
                          init_bias = zeros32)

Soft k-winners-take-all gate via Gumbel-softmax. On each forward in
training mode the layer draws Gumbel noise over its `out` neurons,
adds it to the pre-activation `h = W·x + b`, and scales by the
state's current `temperature`; the softmax of that is the gate. At
high τ the gate is nearly uniform → near-dense representation. At
low τ the gate concentrates on a small subset → sparse
representation. Training can anneal τ from 1.0 → 0.1 to move the
layer from dense to sparse without ever running into the
non-differentiable argmax KATE uses.

Test-mode forward drops the Gumbel noise (deterministic gate = `softmax(h/τ)`).
"""
struct GumbelSoftCompetetive{F, IW, IB} <: AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::F
    init_temperature::Float32
    init_weight::IW
    init_bias::IB
end

function GumbelSoftCompetetive(pair::Pair{<:Integer, <:Integer}, σ = gelu;
                               init_temperature::Real = 1.0f0,
                               init_weight = glorot_uniform,
                               init_bias = zeros32)
    return GumbelSoftCompetetive(Int(pair.first), Int(pair.second),
                                 σ, Float32(init_temperature),
                                 init_weight, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::GumbelSoftCompetetive)
    return (weight = l.init_weight(rng, l.out_dims, l.in_dims),
            bias   = l.init_bias(rng, l.out_dims))
end

function LuxCore.initialstates(rng::AbstractRNG, l::GumbelSoftCompetetive)
    return (rng = Lux.replicate(rng),
            temperature = l.init_temperature,
            training = Val(true))
end

LuxCore.parameterlength(l::GumbelSoftCompetetive) =
    l.out_dims * l.in_dims + l.out_dims
LuxCore.statelength(::GumbelSoftCompetetive) = 0

function Base.show(io::IO, l::GumbelSoftCompetetive)
    print(io, "GumbelSoftCompetetive(", l.in_dims, " => ", l.out_dims)
    l.activation === identity || print(io, ", ", l.activation)
    print(io, "; init_temperature=", l.init_temperature, ")")
end

@inline function _gumbel_noise(rng::AbstractRNG, shape, T::Type)
    # Gumbel(0,1) via the inverse-CDF trick; clamp U to avoid log(0).
    # `@ignore_derivatives` because the noise is a sampling operation
    # we don't want Zygote to differentiate through, and the underlying
    # `rand` / `clamp` trace contains in-place mutations.
    return @ignore_derivatives begin
        U = rand(rng, T, shape...)
        U = clamp.(U, T(1e-9), T(1) - T(1e-9))
        -log.(-log.(U))
    end
end

function (l::GumbelSoftCompetetive)(x::AbstractMatrix, ps, st::NamedTuple)
    T = eltype(x)
    h = ps.weight * x .+ ps.bias
    τ = T(st.temperature)
    gate = if st.training === Val(true)
        noise = _gumbel_noise(st.rng, size(h), T)
        softmax((h .+ noise) ./ τ; dims = 1)
    else
        softmax(h ./ τ; dims = 1)
    end
    # No rescaling: at τ → 0 this approaches hard argmax (gate → one-hot,
    # output = h only at the winner). At higher τ the gate attenuates
    # uniformly. Early experiments with a `* out_dims` factor saturated
    # `tanh` at low τ and collapsed the latent — drop it.
    out = l.activation.(h .* gate)
    # Forward the RNG state so repeated calls draw fresh noise.
    st_new = merge(st, (rng = Lux.replicate(st.rng),))
    return out, st_new
end

function (l::GumbelSoftCompetetive)(x::AbstractVector, ps, st::NamedTuple)
    y, st_new = l(reshape(x, :, 1), ps, st)
    return vec(y), st_new
end

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

"""
    soft_kate(n::Integer;
              hidden::AbstractVector{<:Integer} = [128, 64],
              latent::Integer = 16,
              activation = gelu,
              bottleneck_activation = tanh,
              init_temperature::Real = 1.0f0,
              p::Real = 0.2f0,
              layernorm::Bool = true) -> Chain

Parametric SoftKATE autoencoder. Topology mirrors the parametric
DeepKATE, with four substitutions:

- **GumbelSoftCompetetive** at the bottleneck instead of
  `KCompetetive`.
- **gelu** as the default hidden activation (vs sigmoid/tanh).
- **LayerNorm** after each intermediate Dense (optional, default on).
- **tanh** on the bottleneck output so the latent is bounded without
  the thesis's sine projection.

The encoder ends at the GumbelSoftCompetetive layer; the decoder
mirrors with Dense + activation pairs and a final `Dense → sigmoid`
for BCE-compatible outputs.
"""
function soft_kate(n::Integer;
                   hidden::AbstractVector{<:Integer} = [128, 64],
                   latent::Integer = 16,
                   activation = gelu,
                   init_temperature::Real = 1.0f0,
                   p::Real = 0.2f0,
                   layernorm::Bool = true)
    latent > 0 || throw(ArgumentError("latent must be ≥ 1"))
    hidden = Int[Int(h) for h in hidden]
    any(h -> h <= 0, hidden) && throw(ArgumentError("hidden sizes must be ≥ 1"))

    layers = Any[]
    # --- Encoder ---
    prev = Int(n)
    for h in hidden
        push!(layers, Dense(prev => h, activation))
        layernorm && push!(layers, LayerNorm((h,)))
        p > 0 && push!(layers, Dropout(Float32(p)))
        prev = h
    end
    # Soft competition → bottleneck.
    push!(layers,
          GumbelSoftCompetetive(prev => Int(latent), activation;
                                init_temperature = init_temperature))
    # Use `tanh` as the bottleneck output so the latent is bounded
    # (swap for the thesis's `sin`).
    push!(layers, Dense(Int(latent) => Int(latent), tanh))

    # --- Decoder (mirror) ---
    prev = Int(latent)
    for h in reverse(hidden)
        push!(layers, Dense(prev => h, activation))
        layernorm && push!(layers, LayerNorm((h,)))
        p > 0 && push!(layers, Dropout(Float32(p)))
        prev = h
    end
    push!(layers, Dense(prev => Int(n), sigmoid))

    return Chain(layers...)
end

"""
    latent_layer(model::Chain) -> Int

1-based index of the SoftKATE bottleneck Dense — the one with
`tanh` activation that sits right after `GumbelSoftCompetetive`.
"""
function latent_layer(model::Chain)
    for (i, layer) in enumerate(values(model.layers))
        if layer isa Dense && getproperty(layer, :activation) === tanh
            return i
        end
    end
    throw(ArgumentError("model has no tanh-activated bottleneck Dense; \
                         not built by `soft_kate`?"))
end

# ---------------------------------------------------------------------------
# Temperature annealing
# ---------------------------------------------------------------------------

"""
    set_temperature(st::NamedTuple, τ) -> NamedTuple

Walk the Lux state tree and set `temperature` on every
`GumbelSoftCompetetive` sub-state. Returns a new state (Lux states
are immutable); no side effects on the input. Call once per epoch
(or per step) with the annealed `τ`.
"""
function set_temperature(st::NamedTuple, τ::Real)
    τf = Float32(τ)
    return _map_temperature(st, τf)
end

function _map_temperature(st::NamedTuple, τ::Float32)
    keys_ = keys(st)
    vals = map(keys_) do k
        v = getfield(st, k)
        if k === :temperature && v isa Real
            τ
        elseif v isa NamedTuple
            _map_temperature(v, τ)
        else
            v
        end
    end
    return NamedTuple{keys_}(vals)
end

"""
    anneal_temperature(st, step, total_steps;
                       start = 1.0, stop = 0.1,
                       schedule = :cosine) -> NamedTuple

Convenience: compute the annealed temperature for `step` of
`total_steps` and call [`set_temperature`]. `schedule = :cosine`
(default) follows half a cosine decay from `start` to `stop`;
`:linear` interpolates linearly.
"""
function anneal_temperature(st::NamedTuple,
                            step::Integer, total_steps::Integer;
                            start::Real = 1.0, stop::Real = 0.1,
                            schedule::Symbol = :cosine)
    total_steps >= 1 || throw(ArgumentError("total_steps must be ≥ 1"))
    frac = clamp(Float64(step - 1) / max(1, total_steps - 1), 0.0, 1.0)
    τ = if schedule === :cosine
        stop + (start - stop) * 0.5 * (1 + cos(π * frac))
    elseif schedule === :linear
        start + (stop - start) * frac
    else
        throw(ArgumentError("unknown schedule `$schedule`; use :cosine or :linear"))
    end
    return set_temperature(st, τ), τ
end

# ---------------------------------------------------------------------------
# Loss: joint BCE + SimCSE
# ---------------------------------------------------------------------------

@inline _log_safe(x::T) where {T} = log(max(x, T(1e-7)))

@inline function _bce(ŷ::AbstractArray, y::AbstractArray)
    return -mean(@. y * _log_safe(ŷ) + (1 - y) * _log_safe(1 - ŷ))
end

# Forward up to `upto` (1-based, inclusive) inside the Chain.
function _forward_to(model::Chain, ps, st, x, upto::Int)
    nlayers = length(model.layers)
    out = x
    for i in 1:nlayers
        sym = Symbol(:layer_, i)
        layer = getfield(model.layers, sym)
        p = getfield(ps, sym)
        s = getfield(st, sym)
        out, s_new = layer(out, p, s)
        st = merge(st, NamedTuple{(sym,)}((s_new,)))
        i == upto && return out, st
    end
    return out, st
end

function _forward_from(model::Chain, ps, st, z, from::Int)
    nlayers = length(model.layers)
    out = z
    for i in from:nlayers
        sym = Symbol(:layer_, i)
        layer = getfield(model.layers, sym)
        p = getfield(ps, sym)
        s = getfield(st, sym)
        out, s_new = layer(out, p, s)
        st = merge(st, NamedTuple{(sym,)}((s_new,)))
    end
    return out, st
end

"""
    soft_kate_loss(model, ps, st, x::AbstractMatrix;
                   λ::Real = 0.5, τ_contrast::Real = 0.05) -> (loss, st_new)

Joint reconstruction + contrastive objective. Runs the encoder
twice with fresh Gumbel noise (and fresh Dropout masks) to obtain a
positive-pair `(z, z⁺)` at the bottleneck; runs the decoder once on
`z` to get the reconstruction. The loss is

    loss = BCE(decoder(z), x)  +  λ · SimCSE(z, z⁺; τ_contrast).

`λ = 0` recovers pure BCE; `λ = 1` weighs reconstruction and
contrastive equally. `τ_contrast` is SimCSE's temperature (not the
competitive layer's). Returns the state after both forward passes
so the caller's RNG stays consistent across batches.
"""
function soft_kate_loss(model::Chain, ps, st,
                        x::AbstractMatrix{<:Real};
                        λ::Real = 0.5,
                        τ_contrast::Real = 0.05)
    lat = latent_layer(model)
    # First pass: encoder → z1, decoder → reconstruction.
    z1, st1 = _forward_to(model, ps, st,  x, lat)
    ŷ,  st2 = _forward_from(model, ps, st1, z1, lat + 1)
    # Second pass: encoder only → z2 (fresh Gumbel noise from rng in st2).
    z2, st3 = _forward_to(model, ps, st2, x, lat)
    recon = _bce(ŷ, x)
    contra = simcse_loss(z1, z2; τ = τ_contrast)
    return recon + eltype(ŷ)(λ) * contra, st3
end

end # module SoftKATE
