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
    deep_kate(n::Integer; latent::Integer = 2,
              k1::Integer = 25, p::Real = 0.4f0) -> Chain

Build the DeepKATE autoencoder for input dimension `n` with `latent`
latent dimensions. `k1` is the `k`-winners-take-all count of the first
competitive layer (thesis default 25); the bottleneck competitive layer
has `k = latent` winners. `p` is the dropout rate (thesis default 0.4).

Mirrors Quellcode 3.7 exactly, one layer for one line.
"""
function deep_kate(n::Integer; latent::Integer = 2,
                   k1::Integer = 25, p::Real = 0.4f0)
    return Chain(
        KCompetetive(n, 100, tanh; k = k1),        # encoder
        Dense(100 => 20, sigmoid),
        Dropout(Float32(p)),
        KCompetetive(20, 5, tanh; k = latent),
        Dense(5 => latent, sin),                   # bottleneck
        Dense(latent => 5, sigmoid),               # decoder
        Dense(5 => 20, sigmoid),
        Dropout(Float32(p)),
        Dense(20 => 100, sigmoid),
        Dense(100 => n, sigmoid),
    )
end

"""
    latent_layer(::Chain) -> Int

Index of the last encoder layer in a DeepKATE model. The encoder is
`model[1:latent_layer(model)]`; the decoder is
`model[latent_layer(model)+1:end]`.
"""
latent_layer(::Chain) = 5

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
