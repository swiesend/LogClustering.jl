"""
    DeepKATE

Port of the thesis's DeepKATE autoencoder (§3.2.3, Quellcode 3.7/3.8/3.9)
to Lux. DeepKATE stacks [`KATE.KCompetetive`](@ref) layers inside a deeper
`Chain` to learn a grouping embedding: the competitive layers force a
sparse `k`-winner activation; the sine-activated bottleneck spreads
instances that would otherwise collapse to the same activation magnitude;
dropout before each competitive layer encourages the surviving winners
to encode complex features rather than the obvious ones.

The original thesis model (Flux.jl):

```julia
m = Chain(
    KATE.KCompetetive(n, 100, tanh, k=25),
    Dense(100, 20, sigmoid),
    Dropout(0.4),
    KATE.KCompetetive(20,  5, tanh, k=l),
    Dense( 5,  l, sin),
    Dense( l,  5, sigmoid),
    Dense( 5, 20, sigmoid),
    Dropout(0.4),
    Dense(20, 100, sigmoid),
    Dense(100, n, sigmoid),
)
```

Ported to Lux, [`deep_kate`](@ref) returns exactly this topology with the
KATE competition reused from `src/KATE.jl` (no duplication of the sort).

The encoder output is produced by the sine-activated `Dense` at position
`latent_layer(model)`; the loss [`deep_kate_loss`](@ref) splits the model
at that index to build the triplet Pareto objective (prev / ce / succ)
from Quellcode 3.8.
"""
module DeepKATE

using Lux
using Random
using Statistics
using ChainRulesCore: @ignore_derivatives
using ..KATE: KCompetetive

export deep_kate, latent_layer, deep_kate_loss, repel

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

"""
    deep_kate(n::Integer;
              hidden::AbstractVector{<:Integer} = [100, 20],
              latent::Integer = 2,
              k1::Integer = 25,
              k_bottleneck::Integer = latent,
              p::Real = 0.4f0) -> Chain

Build a DeepKATE autoencoder.

- `hidden` is the encoder's widths from input toward the bottleneck;
  the decoder mirrors it. `[100, 20]` (the default) recovers the
  thesis's semantics.
- `latent` is the bottleneck dimension; it is also the default for
  `k_bottleneck` (k-winners at the bottleneck KATE layer).
- `k1` is the k-winners count at the first (widest) KATE layer.
- `p` is the Dropout rate; one Dropout after each intermediate
  encoder transition (and mirrored in the decoder).

Topology with `hidden = [h₁, h₂, …, hₘ]`:

```
Encoder:
    KCompetetive(n → h₁; k = k1)
    Dense(h₁ → h₂, sigmoid); Dropout(p)
    …
    Dense(hₘ₋₁ → hₘ, sigmoid); Dropout(p)
    KCompetetive(hₘ → latent; k = k_bottleneck)
    Dense(latent → latent, sin)            ← bottleneck output

Decoder (mirror):
    Dense(latent → hₘ, sigmoid)
    Dense(hₘ → hₘ₋₁, sigmoid); Dropout(p)
    …
    Dense(h₂ → h₁, sigmoid); Dropout(p)
    Dense(h₁ → n, sigmoid)
```

`hidden = []` collapses to the minimum topology
`KCompetetive(n → latent) → Dense(latent → latent, sin) →
Dense(latent → n, sigmoid)`.

Unlike the thesis's fixed 10-layer form, this factory scales the
bottleneck with `latent`: on a corpus with many event types
(e.g. Thunderbird's 149) set `hidden = [256, 64]` and
`latent = 32` to avoid the narrow-bottleneck collapse. Thesis
defaults (`hidden = [100, 20]`, `latent = 2`) stay backward-
compatible for API callers who don't opt in.
"""
function deep_kate(n::Integer;
                   hidden::AbstractVector{<:Integer} = [100, 20],
                   latent::Integer = 2,
                   k1::Integer = 25,
                   k_bottleneck::Integer = latent,
                   p::Real = 0.4f0)
    latent > 0 || throw(ArgumentError("latent must be ≥ 1"))
    k1 > 0     || throw(ArgumentError("k1 must be ≥ 1"))
    k_bottleneck > 0 || throw(ArgumentError("k_bottleneck must be ≥ 1"))
    hidden = Int[Int(h) for h in hidden]
    any(h -> h <= 0, hidden) && throw(ArgumentError("hidden sizes must be ≥ 1"))

    layers = Any[]

    # --- Encoder -----------------------------------------------------------
    if isempty(hidden)
        push!(layers, KCompetetive(Int(n), Int(latent), tanh; k = Int(k_bottleneck)))
    else
        k1_eff = min(Int(k1), hidden[1])
        push!(layers, KCompetetive(Int(n), hidden[1], tanh; k = k1_eff))
        for i in 1:length(hidden) - 1
            push!(layers, Dense(hidden[i] => hidden[i + 1], sigmoid))
            push!(layers, Dropout(Float32(p)))
        end
        push!(layers, KCompetetive(hidden[end], Int(latent), tanh;
                                   k = Int(k_bottleneck)))
    end

    # --- Bottleneck --------------------------------------------------------
    push!(layers, Dense(Int(latent) => Int(latent), sin))

    # --- Decoder (mirror) --------------------------------------------------
    if isempty(hidden)
        push!(layers, Dense(Int(latent) => Int(n), sigmoid))
    else
        push!(layers, Dense(Int(latent) => hidden[end], sigmoid))
        for i in length(hidden):-1:2
            push!(layers, Dense(hidden[i] => hidden[i - 1], sigmoid))
            push!(layers, Dropout(Float32(p)))
        end
        push!(layers, Dense(hidden[1] => Int(n), sigmoid))
    end

    return Chain(layers...)
end

"""
    latent_layer(model::Chain) -> Int

1-based index of the DeepKATE bottleneck (the sin-activated Dense).
The encoder is `model[1:latent_layer(model)]`; the decoder is
`model[latent_layer(model)+1:end]`. Under the parametric factory
this index moves with `length(hidden)` (`5` for the thesis default
`hidden = [100, 20]`; `7` for `hidden = [256, 128, 64]`; etc.).
"""
function latent_layer(model::Chain)
    for (i, layer) in enumerate(values(model.layers))
        if layer isa Dense && getproperty(layer, :activation) === sin
            return i
        end
    end
    throw(ArgumentError("model has no sin-activated bottleneck Dense; \
                         not built by `deep_kate`?"))
end

# ---------------------------------------------------------------------------
# Repel helper (Quellcode 3.9)
# ---------------------------------------------------------------------------

"""
    repel(xs; pivot = 0) -> Array

Step-function companion to the sine activation: each component of `xs`
above `pivot` is pushed to the next integer up, each below to the
previous integer down. Its negation `-repel(xs)` is used as the training
target for the latent embedding in the prev/succ sub-objectives
(Quellcode 3.8). Non-mutating; Zygote-friendly.
"""
function repel(xs::AbstractArray{T}; pivot::Real = zero(T)) where {T<:Real}
    p = T(pivot)
    return @. ifelse(xs >= p, ceil(xs), floor(xs))
end

# ---------------------------------------------------------------------------
# Loss (Quellcode 3.8)
# ---------------------------------------------------------------------------

@inline _log_safe(x::T) where {T} = log(max(x, T(1e-7)))

"""
    deep_kate_loss(model, ps, st, a, b, c; lat = latent_layer(model)) -> (loss, st)

Pareto objective from thesis Algorithm 3.8:

  `ce`   — binary cross-entropy between the reconstruction of `b` and `b`
           itself (the standard AE signal),
  `prev` — MSE between `b`'s latent embedding and `-target_a`, where
           `target_a` is `a`'s latent embedding under a *test-mode* copy
           of the encoder (no KATE competition, no dropout),
  `succ` — same against `c`.

`prev + ce + succ` is returned. Inputs `a`, `b`, `c` are `(n, batch)`
arrays (or vectors). The forward pass is written out layer-by-layer
instead of via `Chain(...)` so Zygote doesn't trip on the chain
constructor during reverse-mode AD.
"""
function deep_kate_loss(
    model::Chain, ps, st,
    a::AbstractArray{<:Real}, b::AbstractArray{<:Real}, c::AbstractArray{<:Real};
    lat::Int = latent_layer(model),
)
    nlayers = length(model.layers)
    # Targets are stop-gradient: the thesis uses `Flux.data(...)` to strip
    # the tracked grads of `target_a` and `target_c`, so only the
    # `embedded` path contributes to `∂loss/∂ps`.
    target_a = @ignore_derivatives _fwd(model, ps, Lux.testmode(st), a, 1, lat)
    target_c = @ignore_derivatives _fwd(model, ps, Lux.testmode(st), c, 1, lat)
    embedded = _fwd(model, ps, st, b, 1, lat)
    prediction = _fwd(model, ps, st, embedded, lat + 1, nlayers)

    prev = _mse(embedded, -target_a)
    succ = _mse(embedded, -target_c)
    ce = _binary_crossentropy(prediction, b)
    return prev + ce + succ, st
end

"""
    deep_kate_loss(model, ps, st, X::AbstractMatrix; lat = latent_layer(model))
        -> (loss, st)

Single-batch overload. Drops the thesis's `prev`/`succ` temporal
sub-objectives and returns **only** the binary-cross-entropy
reconstruction of `X` — the standard autoencoder signal.

Use this when you're training on a shuffled batch without access
to each sample's event-stream neighbours (e.g. BoW over a whole
dataset, the case the CLI's `train --kind deep_kate` drives).
For a time-ordered trace where `prev` / `succ` are meaningful,
call the six-argument form `deep_kate_loss(model, ps, st, a, b, c)`
directly.

The returned `st` is the state from the reconstruction pass; no
stop-gradient book-keeping happens here (there are no ignored
target branches).
"""
function deep_kate_loss(
    model::Chain, ps, st,
    X::AbstractArray{<:Real};
    lat::Int = latent_layer(model),
)
    nlayers = length(model.layers)
    embedded = _fwd(model, ps, st, X, 1, lat)
    prediction = _fwd(model, ps, st, embedded, lat + 1, nlayers)
    ce = _binary_crossentropy(prediction, X)
    return ce, st
end

# Walk the NamedTuple of layers over a contiguous index range, calling
# each layer's functor form. Returned state is discarded (the thesis
# loop re-runs setup each epoch); we only need the forward value for AD.
@generated function _fwd(model::Chain, ps, st, x, from::Int, to::Int)
    :(begin
        out = x
        i = from
        while i <= to
            sym = _layer_sym(i)
            layer = getfield(model.layers, sym)
            p = getfield(ps, sym)
            s = getfield(st, sym)
            out, _ = layer(out, p, s)
            i += 1
        end
        out
    end)
end

@inline _layer_sym(i::Int) =
    i == 1  ? :layer_1  : i == 2  ? :layer_2  :
    i == 3  ? :layer_3  : i == 4  ? :layer_4  :
    i == 5  ? :layer_5  : i == 6  ? :layer_6  :
    i == 7  ? :layer_7  : i == 8  ? :layer_8  :
    i == 9  ? :layer_9  : i == 10 ? :layer_10 :
    Symbol(:layer_, i)

@inline _mse(x, y) = mean(abs2, x .- y)

@inline function _binary_crossentropy(ŷ::AbstractArray, y::AbstractArray)
    # Matches Flux.crossentropy's semantics for the binary case used in
    # the thesis: negative mean log-likelihood of the target.
    return -mean(@. y * _log_safe(ŷ) + (1 - y) * _log_safe(1 - ŷ))
end

end # module DeepKATE
