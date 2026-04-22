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

## Annealing caveat

The default annealing target is `τ_stop ≈ 1.0` rather than the
"hard" value `τ ≈ 0.1`. Empirically on Thunderbird_2k (149
templates into 40 latent dims), driving τ all the way to `0.1-0.3`
collapses sparsity-based clustering: the gate becomes a near
one-hot k-WTA, and distinct templates get forced onto the same
winner index. Rule of thumb:

    latent_dim < num_templates  →  keep τ_stop ≥ 1.0
    latent_dim ≥ num_templates  →  safe to anneal to τ_stop ≈ 0.1

The 240-epoch Thunderbird sweep measured sparsity-top-5 purity
peaking at **τ = 1.75 / epoch 120** (Pur 0.849, NMI 0.893), and
dropping back to 0.833 at epoch 240 as τ continued annealing.
Hold τ in the soft regime unless you know the bottleneck has
headroom.

## Training helper

Use [`soft_kate_train!`] when you want the loop + annealing +
optional best-checkpoint tracking against a held-out sample. The
manual-loop form stays available for full control.
"""
module SoftKATE

using Lux
using LuxCore: LuxCore, AbstractLuxLayer
using Random: AbstractRNG, default_rng, MersenneTwister, shuffle
using Statistics: mean
using NNlib: softmax, logsoftmax, gelu, sigmoid
using WeightInitializers: glorot_uniform, zeros32
using ChainRulesCore: @ignore_derivatives
using Zygote
using Clustering: kmeans
using ..SimCSE: simcse_loss
using ..Sparsity: sparsity_clusters
using ..Pipeline: l2_normalise
using ..Metrics: nmi, ari, purity

export GumbelSoftCompetetive, soft_kate, soft_kate_loss, latent_layer,
       set_temperature, anneal_temperature, soft_kate_train!

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

# ---------------------------------------------------------------------------
# Training helper with optional best-checkpoint tracking
# ---------------------------------------------------------------------------

@inline function _sgd_step(ps, grads, lr)
    grads === nothing && return ps
    ps isa AbstractArray && return ps .- lr .* grads
    if ps isa NamedTuple
        k = keys(ps)
        return NamedTuple{k}(map(ki -> _sgd_step(getfield(ps, ki),
                                                 hasproperty(grads, ki) ? getfield(grads, ki) : nothing,
                                                 lr), k))
    end
    return ps
end

function _encode(model::Chain, ps, st, X::AbstractMatrix, lat::Int)
    st_eval = Lux.testmode(st)
    Z = X
    for i in 1:lat
        sym = Symbol(:layer_, i)
        Z, _ = getfield(model.layers, sym)(Z, getfield(ps, sym),
                                           getfield(st_eval, sym))
    end
    return Z
end

"""
    _validate(model, ps, st, Xval, y_gt; cluster, sparsity_k, kmeans_k, metric) -> Float64

Run the encoder on `Xval`, cluster the L2-normalised latent, score
the clustering against `y_gt` (string labels or integer ids) with
the chosen metric. `cluster = :sparsity` uses `sparsity_clusters`
with `sparsity_k`; `:kmeans` uses `kmeans_cluster` with `kmeans_k`.
"""
function _validate(model::Chain, ps, st,
                   Xval::AbstractMatrix, y_gt::AbstractVector;
                   cluster::Symbol, sparsity_k::Integer,
                   kmeans_k::Integer, metric::Symbol)
    lat = latent_layer(model)
    Z = _encode(model, ps, st, Xval, lat)
    Zn = l2_normalise(Z)
    preds = if cluster === :sparsity
        first(sparsity_clusters(Zn, Int(sparsity_k)))
    elseif cluster === :kmeans
        km = kmeans(Zn, Int(kmeans_k); maxiter = 200)
        km.assignments
    else
        throw(ArgumentError("unknown cluster strategy `$cluster`; use :sparsity or :kmeans"))
    end
    pred_str = string.(preds)
    gold_str = [String(y) for y in y_gt]
    score = if metric === :nmi
        nmi(pred_str, gold_str)
    elseif metric === :ari
        ari(pred_str, gold_str)
    elseif metric === :purity
        purity(pred_str, gold_str)
    else
        throw(ArgumentError("unknown metric `$metric`; use :nmi, :ari, :purity"))
    end
    return Float64(score)
end

"""
    soft_kate_train!(model::Chain, ps, st, X::AbstractMatrix;
                     epochs = 120, batch = 64, lr = 0.02f0,
                     λ = 0.3, τ_start = 2.0, τ_stop = 1.0,
                     schedule = :cosine,
                     validation = nothing,
                     metric = :ari,
                     cluster = :sparsity,
                     sparsity_k = 5,
                     kmeans_k = 0,
                     patience = 0,
                     rng = MersenneTwister(0),
                     verbose = false)
        -> (ps_best, st_best, history)

Train a SoftKATE `model` on columnar-sample matrix `X` with the
joint BCE + SimCSE objective, annealed Gumbel temperature, and —
if `validation = (Xval, y_gt)` is supplied — best-checkpoint
tracking against a held-out sample.

Knobs:

- `epochs`, `batch`, `lr`: standard SGD.
- `λ`: weight of the SimCSE contrastive term in [`soft_kate_loss`].
- `τ_start`, `τ_stop`, `schedule`: drive
  [`anneal_temperature`] once per gradient step. `τ_stop = 1.0`
  is the default *after* the 240-epoch Thunderbird finding —
  annealing into the hard-kWTA regime collapses sparsity
  clustering when `latent_dim < num_templates`.
- `validation`: optional `(Xval, y_gt)` tuple. `Xval` is a
  feature matrix matching `X`'s layout; `y_gt` is a vector of
  string or integer ground-truth labels.
- `metric`: `:nmi | :ari | :purity` — the scalar tracked against
  `validation`.
- `cluster`: `:sparsity` (default) uses `sparsity_clusters(Z,
  sparsity_k)`; `:kmeans` uses `kmeans(Z, kmeans_k)`.
- `patience = 0`: no early stop (run all epochs, restore best);
  `patience > 0`: early-stop when the validation metric hasn't
  improved for that many epochs.
- `verbose`: print a short per-epoch log to `stderr`.

Returns `(ps_best, st_best, history)`. `history` is a
`Vector{NamedTuple}` with `(epoch, loss, τ, val)` per epoch (`val`
is the validation metric or `NaN` when no validation supplied).

When no `validation` is supplied, `ps_best` / `st_best` are the
final-epoch parameters and no early stopping applies.
"""
function soft_kate_train!(model::Chain, ps, st, X::AbstractMatrix{<:Real};
                          epochs::Integer = 120,
                          batch::Integer = 64,
                          lr::Real = 0.02f0,
                          λ::Real = 0.3,
                          τ_start::Real = 2.0,
                          τ_stop::Real = 1.0,
                          schedule::Symbol = :cosine,
                          validation::Union{Nothing, Tuple} = nothing,
                          metric::Symbol = :ari,
                          cluster::Symbol = :sparsity,
                          sparsity_k::Integer = 5,
                          kmeans_k::Integer = 0,
                          patience::Integer = 0,
                          rng::AbstractRNG = MersenneTwister(0),
                          verbose::Bool = false)
    epochs >= 1 || throw(ArgumentError("epochs must be ≥ 1"))
    N = size(X, 2)
    b = min(Int(batch), N)
    total_steps = Int(epochs) * cld(N, b)
    step = 0

    history = @NamedTuple{epoch::Int, loss::Float64,
                          τ::Float64, val::Float64}[]
    best_val = -Inf
    best_ps, best_st = ps, st
    epochs_since_improve = 0

    if validation !== nothing && cluster === :kmeans && kmeans_k == 0
        throw(ArgumentError("cluster = :kmeans requires kmeans_k > 0"))
    end

    lr_f = Float32(lr)
    for e in 1:Int(epochs)
        perm = shuffle(rng, collect(1:N))
        ep_loss = 0.0f0
        ep_seen = 0
        for start in 1:b:N
            stop = min(start + b - 1, N)
            xb = X[:, perm[start:stop]]
            step += 1
            st, _ = anneal_temperature(st, step, total_steps;
                                       start = τ_start, stop = τ_stop,
                                       schedule = schedule)
            (loss, st), back = Zygote.pullback(
                p -> soft_kate_loss(model, p, st, xb; λ = λ), ps)
            g = back((one(loss), nothing))[1]
            ps = _sgd_step(ps, g, lr_f)
            ep_loss += Float32(loss) * size(xb, 2)
            ep_seen += size(xb, 2)
        end
        epoch_loss = Float64(ep_loss / max(1, ep_seen))
        τ_now = Float64(_find_temperature(st))
        val_score = if validation === nothing
            NaN
        else
            Xval, y_gt = validation
            _validate(model, ps, st, Xval, y_gt;
                      cluster = cluster, sparsity_k = Int(sparsity_k),
                      kmeans_k = Int(kmeans_k), metric = metric)
        end
        push!(history, (epoch = e, loss = epoch_loss, τ = τ_now,
                        val = val_score))
        if validation !== nothing && val_score > best_val
            best_val = val_score
            best_ps, best_st = ps, st
            epochs_since_improve = 0
        else
            epochs_since_improve += 1
        end
        if verbose
            println(stderr, "soft_kate  epoch ", lpad(e, 4), "/",
                    epochs, "   loss=", round(epoch_loss; digits = 4),
                    "   τ=", round(τ_now; digits = 3),
                    validation === nothing ? "" :
                        "   val($metric)=" * string(round(val_score; digits = 4)),
                    "   best=", round(best_val; digits = 4))
        end
        if patience > 0 && validation !== nothing &&
                epochs_since_improve >= Int(patience)
            verbose && println(stderr, "early stop at epoch $e (no improvement for $patience epochs)")
            break
        end
    end
    if validation === nothing
        best_ps, best_st = ps, st
    end
    return best_ps, best_st, history
end

# Walk the Lux state tree to find the first `temperature` field — used
# for logging. O(#sub-states); fine for our small chains.
function _find_temperature(st)
    st isa NamedTuple || return NaN
    if haskey(st, :temperature)
        return Float64(st.temperature)
    end
    for k in keys(st)
        v = getfield(st, k)
        if v isa NamedTuple
            τ = _find_temperature(v)
            isnan(τ) || return τ
        end
    end
    return NaN
end

end # module SoftKATE
