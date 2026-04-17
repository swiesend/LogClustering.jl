"""
    PersistenceGlue

Type-dispatched `save` / `rehydrate` for every persistable artifact in
the package. Lives in its own file so the model modules stay free of
JLD2 references and no one module has to know about the others.

Registration happens from `LogClustering.__init__`, which is where the
per-kind rehydrators land in `Persistence.REHYDRATORS` and the
named callables (Drain's `parametrize`, etc.) land in
`Persistence.CALLABLE_REGISTRY`.
"""
module PersistenceGlue

using ..Persistence
using ..KATE: KATE, KCompetetive
using ..DeepKATE: DeepKATE, deep_kate, latent_layer
using ..SeqLSTM: SeqLSTM, seq_lstm, PeepholeLSTM
using ..VQVAE: VQVAE, vq_vae, VectorQuantizer
using ..Drain3: Drain3, Drain, default_parametrize, TreeNode, LogCluster
using ..Instance: Instance, ValueNoveltyDetector
using ..Dedup: Dedup, DedupState
using ..Masking: Masking, SlotValue
using Lux: Lux, Chain

export save, name_of_activation, activation_by_name

# ---------------------------------------------------------------------------
# Activation-function registry — the one spot where a symbol round-trips
# to a live Lux/Base/Base-in-Lux activation function.
# ---------------------------------------------------------------------------

const _ACTIVATION_BY_NAME = Dict{Symbol, Any}(
    :identity => identity,
    :tanh     => tanh,
    :sigmoid  => Lux.sigmoid,
    :relu     => Lux.relu,
    :sin      => sin,
    :cos      => cos,
)

"Look up an activation function by registered name. Unknown → `identity`."
activation_by_name(sym::Symbol) = get(_ACTIVATION_BY_NAME, sym, identity)

"Inverse of [`activation_by_name`]; returns `:identity` if the function
isn't registered (the round-trip loses the original activation, but
architecture + parameters are preserved)."
function name_of_activation(σ)
    for (n, f) in _ACTIVATION_BY_NAME
        f === σ && return n
    end
    return :identity
end

# ---------------------------------------------------------------------------
# KATE
# ---------------------------------------------------------------------------

function save_kate(path::AbstractString, layer::KCompetetive, ps, st;
                   metadata::AbstractDict = Dict{String, Any}())
    spec = (
        in_dims  = layer.in_dims,
        out_dims = layer.out_dims,
        k        = layer.k,
        activation = name_of_activation(layer.activation),
        alpha    = layer.alpha,
    )
    Persistence.save_lux(path; kind = :kate, spec = spec,
                         ps = ps, st = st, metadata = metadata)
end

function _rehydrate_kate(bundle)
    sp = bundle.spec
    layer = KCompetetive(sp.in_dims, sp.out_dims,
                         activation_by_name(sp.activation), sp.alpha;
                         k = sp.k)
    return (; layer = layer,
              ps = bundle.payload.ps,
              st = bundle.payload.st)
end

# ---------------------------------------------------------------------------
# DeepKATE — constructor kwargs are `(n, latent, k1, p)`.
# ---------------------------------------------------------------------------

function save_deep_kate(path, model::Chain, ps, st;
                        n::Integer, latent::Integer, k1::Integer, p::Real,
                        metadata::AbstractDict = Dict{String, Any}())
    spec = (n = Int(n), latent = Int(latent), k1 = Int(k1), p = Float32(p))
    Persistence.save_lux(path; kind = :deep_kate, spec = spec,
                         ps = ps, st = st, metadata = metadata)
end

function _rehydrate_deep_kate(bundle)
    sp = bundle.spec
    model = deep_kate(sp.n; latent = sp.latent, k1 = sp.k1, p = sp.p)
    return (; model = model,
              ps = bundle.payload.ps,
              st = bundle.payload.st)
end

# ---------------------------------------------------------------------------
# SeqLSTM — plain / peephole / bidirectional flags survive via the spec.
# ---------------------------------------------------------------------------

function save_seq_lstm(path, model::Chain, ps, st;
                       vocab_size::Integer, embed::Integer, hidden::Integer,
                       bidirectional::Bool = false, peephole::Bool = false,
                       metadata::AbstractDict = Dict{String, Any}())
    spec = (
        vocab_size   = Int(vocab_size),
        embed        = Int(embed),
        hidden       = Int(hidden),
        bidirectional = bidirectional,
        peephole     = peephole,
    )
    Persistence.save_lux(path; kind = :seq_lstm, spec = spec,
                         ps = ps, st = st, metadata = metadata)
end

function _rehydrate_seq_lstm(bundle)
    sp = bundle.spec
    model = seq_lstm(sp.vocab_size;
                     embed = sp.embed,
                     hidden = sp.hidden,
                     bidirectional = sp.bidirectional,
                     peephole = sp.peephole)
    return (; model = model,
              ps = bundle.payload.ps,
              st = bundle.payload.st)
end

# ---------------------------------------------------------------------------
# VQ-VAE — factory kwargs are `(n, codebook_size, embed_dim, hidden)`.
# ---------------------------------------------------------------------------

function save_vq_vae(path, model::Chain, ps, st;
                     n::Integer, codebook_size::Integer,
                     embed_dim::Integer, hidden::Integer,
                     metadata::AbstractDict = Dict{String, Any}())
    spec = (
        n             = Int(n),
        codebook_size = Int(codebook_size),
        embed_dim     = Int(embed_dim),
        hidden        = Int(hidden),
    )
    Persistence.save_lux(path; kind = :vq_vae, spec = spec,
                         ps = ps, st = st, metadata = metadata)
end

function _rehydrate_vq_vae(bundle)
    sp = bundle.spec
    model = vq_vae(sp.n;
                   codebook_size = sp.codebook_size,
                   embed_dim     = sp.embed_dim,
                   hidden        = sp.hidden)
    return (; model = model,
              ps = bundle.payload.ps,
              st = bundle.payload.st)
end

# ---------------------------------------------------------------------------
# Drain3 — the one case where a knob is a callable.
# ---------------------------------------------------------------------------

function save_drain(path, d::Drain;
                    metadata::AbstractDict = Dict{String, Any}())
    param_name = Persistence.name_of_callable(d.parametrize)
    spec = (
        depth            = d.depth,
        sim_th           = d.sim_th,
        max_children     = d.max_children,
        max_clusters     = d.max_clusters,
        wildcard         = d.wildcard,
        parametrize_name = param_name,
    )
    payload = (root = d.root, clusters = d.clusters, next_id = d.next_id)
    Persistence.save_native(path; kind = :drain, spec = spec,
                            payload = payload, metadata = metadata)
end

function _rehydrate_drain(bundle)
    sp = bundle.spec
    fn = get(Persistence.CALLABLE_REGISTRY, sp.parametrize_name, nothing)
    fn === nothing && error("""
        Drain bundle references parametrize `$(sp.parametrize_name)`
        which isn't in Persistence.CALLABLE_REGISTRY. Register via
        `Persistence.register_callable!(:name, fn)` before loading.""")
    d = Drain(; depth = sp.depth, sim_th = sp.sim_th,
               max_children = sp.max_children,
               max_clusters = sp.max_clusters,
               wildcard = sp.wildcard,
               parametrize = fn)
    d.root = bundle.payload.root
    d.clusters = bundle.payload.clusters
    d.next_id = bundle.payload.next_id
    return d
end

# ---------------------------------------------------------------------------
# ValueNoveltyDetector + DedupState — the whole mutable struct serialises.
# ---------------------------------------------------------------------------

function save_value_novelty(path, d::ValueNoveltyDetector;
                            metadata::AbstractDict = Dict{String, Any}())
    payload = (
        seen      = d.seen,
        counts    = d.counts,
        sum       = d.sum,
        sumsq     = d.sumsq,
        n_numeric = d.n_numeric,
    )
    Persistence.save_native(path; kind = :value_novelty, spec = NamedTuple(),
                            payload = payload, metadata = metadata)
end

function _rehydrate_value_novelty(bundle)
    p = bundle.payload
    d = ValueNoveltyDetector()
    merge!(d.seen, p.seen)
    merge!(d.counts, p.counts)
    merge!(d.sum, p.sum)
    merge!(d.sumsq, p.sumsq)
    merge!(d.n_numeric, p.n_numeric)
    return d
end

function save_dedup(path, s::DedupState;
                    metadata::AbstractDict = Dict{String, Any}())
    spec = (m = s.m, k = s.k, observed = s.observed)
    payload = (bits = collect(s.bits),)
    Persistence.save_native(path; kind = :dedup, spec = spec,
                            payload = payload, metadata = metadata)
end

function _rehydrate_dedup(bundle)
    sp = bundle.spec
    # DedupState(; expected_n, fpr) picks m/k; we override them from spec.
    s = DedupState(m = sp.m, k = sp.k,
                   bits = BitVector(bundle.payload.bits),
                   observed = sp.observed)
    return s
end

# The DedupState constructor expects kwargs (expected_n, fpr); add a
# field-wise inner constructor over the declared fields so rehydration
# doesn't have to go through the sizing formulas.
function __init__() end

# ---------------------------------------------------------------------------
# Dispatch: register everything with Persistence.
# ---------------------------------------------------------------------------

function register_all!()
    Persistence.register_rehydrator!(:kate,          _rehydrate_kate)
    Persistence.register_rehydrator!(:deep_kate,     _rehydrate_deep_kate)
    Persistence.register_rehydrator!(:seq_lstm,      _rehydrate_seq_lstm)
    Persistence.register_rehydrator!(:vq_vae,        _rehydrate_vq_vae)
    Persistence.register_rehydrator!(:drain,         _rehydrate_drain)
    Persistence.register_rehydrator!(:value_novelty, _rehydrate_value_novelty)
    Persistence.register_rehydrator!(:dedup,         _rehydrate_dedup)
    # Callable registry: Drain's default parametrize + an explicit
    # "match nothing" helper useful for tests.
    Persistence.register_callable!(:digit, default_parametrize)
    Persistence.register_callable!(:none,  _ -> false)
    return nothing
end

# ---------------------------------------------------------------------------
# Type-dispatched `save` — one user-facing entrypoint.
# ---------------------------------------------------------------------------

"""
    save(path, artifact; kwargs...)

Type-dispatched save. The Lux variants (`deep_kate`, `vq_vae`,
`seq_lstm`, `kate`) take `(model, ps, st; spec-kwargs...)`; the
native structs (`Drain`, `ValueNoveltyDetector`, `DedupState`) take
just the struct. Every variant forwards `metadata` to the bundle.

```julia
save(path, d::Drain; metadata = Dict("dataset" => "HDFS_2k"))
save(path, deep_kate_model, ps, st;
     n = 16, latent = 2, k1 = 4, p = 0.4f0)
```
"""
function save end

save(path, d::Drain; kwargs...)                 = save_drain(path, d; kwargs...)
save(path, d::ValueNoveltyDetector; kwargs...)  = save_value_novelty(path, d; kwargs...)
save(path, d::DedupState; kwargs...)            = save_dedup(path, d; kwargs...)

# Lux entries are keyed by the factory's `kind` kwarg, since a bare
# `Chain` doesn't carry its own factory identity.
function save(path, model::Chain, ps, st;
              kind::Symbol, kwargs...)
    kind === :deep_kate && return save_deep_kate(path, model, ps, st; kwargs...)
    kind === :seq_lstm  && return save_seq_lstm(path, model, ps, st; kwargs...)
    kind === :vq_vae    && return save_vq_vae(path, model, ps, st; kwargs...)
    error("unknown Lux kind `$kind`. Use :deep_kate, :seq_lstm, or :vq_vae.")
end

function save(path, layer::KCompetetive, ps, st; kwargs...)
    save_kate(path, layer, ps, st; kwargs...)
end

end # module PersistenceGlue
