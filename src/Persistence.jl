"""
    Persistence

One save/load API across every trained artifact in the repo — Lux
autoencoders, the SeqLSTM + peephole variants, Drain3, and the
stateful detectors (`ValueNoveltyDetector`, `DedupState`). Uses
JLD2 as the backing format.

Bundle layout:

```julia
struct PersistedBundle
    kind::Symbol                  # :deep_kate, :vq_vae, :simcse_encoder,
                                  # :seq_lstm, :kate, :drain,
                                  # :value_novelty, :dedup, :chain
    lc_version::VersionNumber     # LogClustering.jl version at save time
    schema_version::UInt8         # bumped on bundle-layout break
    spec::NamedTuple              # constructor kwargs; re-runs the factory
    payload::Any                  # (ps, st) for Lux models, struct
                                  # fields for native detectors
    metadata::Dict{String, Any}   # free-form — dataset, date, notes
end
```

Public API:

```julia
save(path, artifact; metadata = Dict())       # type-dispatched
load(path) -> PersistedBundle
rehydrate(bundle) -> rebuilt artifact
```

Each model module registers a rehydrator on load via
[`register_rehydrator!`]; unrecognised `kind`s error with a clear
message. Callables that can't be serialised (e.g. Drain's
`parametrize`) route through [`register_callable!`] + a symbol
stored in the bundle's `spec`.
"""
module Persistence

using JLD2
using SHA: sha256

export PersistedBundle, save, load, rehydrate,
       register_rehydrator!, register_callable!,
       corpus_fingerprint, list_bundles, find_compatible_bundle

const SCHEMA_VERSION = UInt8(2)

"""
    LC_VERSION[]

The `VersionNumber` stamped into every bundle at save time. Populated
lazily from `Project.toml` on first access so the constant tracks the
package's current version without hard-coding or pulling in `Pkg`.
"""
const LC_VERSION = Ref{VersionNumber}(v"0.0.0")

function _lc_version()
    if LC_VERSION[] == v"0.0.0"
        try
            toml = joinpath(@__DIR__, "..", "Project.toml")
            for line in eachline(toml)
                m = match(r"^version\s*=\s*\"([^\"]+)\"", line)
                if m !== nothing
                    LC_VERSION[] = VersionNumber(m.captures[1])
                    break
                end
            end
        catch
            LC_VERSION[] = v"0.0.0"
        end
    end
    return LC_VERSION[]
end

"""
    PersistedBundle(kind, lc_version, schema_version, spec, payload, metadata)

The on-disk representation of a saved artifact. Construct via `save`
or `load`; consumers rarely build one by hand.
"""
struct PersistedBundle
    kind::Symbol
    lc_version::VersionNumber
    schema_version::UInt8
    spec::NamedTuple
    payload::Any
    metadata::Dict{String, Any}
end

Base.show(io::IO, b::PersistedBundle) =
    print(io, "PersistedBundle(", b.kind, ", v", b.lc_version,
          ", schema=", Int(b.schema_version),
          ", metadata=", length(b.metadata), " keys)")

# ---------------------------------------------------------------------------
# Registries — populated by each model module's __init__.
# ---------------------------------------------------------------------------

"Registered rehydrators, one per `kind`. Built by model-module `__init__`."
const REHYDRATORS = Dict{Symbol, Function}()

"""
    register_rehydrator!(kind::Symbol, fn)

Register `fn(bundle) -> artifact` as the rehydrator for `kind`. The
callable must accept a `PersistedBundle` and return whatever
representation the caller expects (often `(model, ps, st)` for Lux
models, or the native struct for detectors).
"""
function register_rehydrator!(kind::Symbol, fn)
    REHYDRATORS[kind] = fn
    return fn
end

"""
Registered callables — named closures used as model knobs (e.g.
Drain's `parametrize`). On save, we store the *name*; on load, we
look it up here. Custom predicates round-trip only if registered.
"""
const CALLABLE_REGISTRY = Dict{Symbol, Function}()

"""
    register_callable!(name::Symbol, fn)

Make `fn` retrievable by `name` during rehydration. Pair with a
`name_of_callable(fn)` lookup for save-time inversion.
"""
function register_callable!(name::Symbol, fn)
    CALLABLE_REGISTRY[name] = fn
    return fn
end

"""
    name_of_callable(fn) -> Symbol

Inverse of the callable registry. Returns `:custom` when `fn` is not
registered — which means the bundle will round-trip only if the
user re-registers the same `fn` under the same name before loading.
"""
function name_of_callable(fn)
    for (name, f) in CALLABLE_REGISTRY
        f === fn && return name
    end
    return :custom
end

# ---------------------------------------------------------------------------
# Generic save/load
# ---------------------------------------------------------------------------

function _save_bundle(path::AbstractString, bundle::PersistedBundle)
    jldsave(path;
        kind = bundle.kind,
        lc_version = string(bundle.lc_version),
        schema_version = bundle.schema_version,
        spec = bundle.spec,
        payload = bundle.payload,
        metadata = bundle.metadata,
    )
    return nothing
end

"""
    load(path) -> PersistedBundle

Reconstruct the bundle from disk. Raises on schema mismatch with a
remediation hint.
"""
function load(path::AbstractString)::PersistedBundle
    isfile(path) || throw(ArgumentError("no file at $path"))
    kind, lc, schema, spec, payload, metadata = jldopen(path, "r") do f
        (f["kind"], f["lc_version"], f["schema_version"],
         f["spec"], f["payload"], f["metadata"])
    end
    schema > SCHEMA_VERSION && error("""
        Bundle schema version $schema is newer than this build
        ($(Int(SCHEMA_VERSION))). Upgrade LogClustering.jl to load.""")
    return PersistedBundle(
        Symbol(kind),
        VersionNumber(lc),
        UInt8(schema),
        spec,
        payload,
        Dict{String, Any}(string(k) => v for (k, v) in metadata),
    )
end

"""
    rehydrate(bundle::PersistedBundle) -> artifact

Dispatch on `bundle.kind` to the registered rehydrator. Errors with
a clear message when the kind is unknown — which usually means the
relevant model module hasn't been loaded (e.g. loading a `:vq_vae`
bundle without `using LogClustering.VQVAE` in the session).
"""
function rehydrate(bundle::PersistedBundle)
    fn = get(REHYDRATORS, bundle.kind, nothing)
    fn === nothing && error("""
        No rehydrator for kind $(bundle.kind). Make sure the
        relevant model module is loaded; registered kinds:
        $(sort(collect(keys(REHYDRATORS))))""")
    return fn(bundle)
end

"""
    load_and_rehydrate(path) -> artifact

Convenience over `rehydrate(load(path))`.
"""
load_and_rehydrate(path::AbstractString) = rehydrate(load(path))

export load_and_rehydrate

# ---------------------------------------------------------------------------
# Lux-model save surface — one entry point, dispatch on `kind`.
# ---------------------------------------------------------------------------

"""
    save_lux(path; kind, spec, ps, st, metadata = Dict())

Save any Lux model via its factory kwargs (`spec`) and trained
parameters / state. The per-kind rehydrator calls the factory with
`spec...` and restores `(ps, st)`.

Most callers use the `save(path, model; …)` wrappers in each model
module instead — this is the low-level path those wrappers route
through.
"""
function save_lux(path::AbstractString;
                  kind::Symbol,
                  spec::NamedTuple,
                  ps,
                  st,
                  metadata::AbstractDict = Dict{String, Any}())
    bundle = PersistedBundle(
        kind, _lc_version(), SCHEMA_VERSION,
        spec, (ps = ps, st = st),
        Dict{String, Any}(string(k) => v for (k, v) in metadata),
    )
    _save_bundle(path, bundle)
end

"""
    save_native(path; kind, spec, payload, metadata = Dict())

Save a native Julia struct / detector. `spec` captures constructor
kwargs; `payload` captures the runtime state (struct fields, tree
nodes, etc.). The per-kind rehydrator rebuilds from both.
"""
function save_native(path::AbstractString;
                     kind::Symbol,
                     spec::NamedTuple,
                     payload,
                     metadata::AbstractDict = Dict{String, Any}())
    bundle = PersistedBundle(
        kind, _lc_version(), SCHEMA_VERSION,
        spec, payload,
        Dict{String, Any}(string(k) => v for (k, v) in metadata),
    )
    _save_bundle(path, bundle)
end

# ---------------------------------------------------------------------------
# Corpus fingerprint + model registry (plan: parametric DeepKATE + reuse)
# ---------------------------------------------------------------------------
#
# A fingerprint binds a saved model to the *tokeniser* it was trained on.
# Two models with the same fingerprint can classify lines from the same
# tokenisation regime; different fingerprints mean the vocab is
# structurally different and reuse would produce garbage.

"""
    corpus_fingerprint(tokens::AbstractVector{<:AbstractString};
                       n_lines::Integer = 0) -> String

Return a hex-encoded SHA-256 of the tokeniser's vocabulary. Stable
across runs and Julia versions; changes whenever a token is added,
removed or renamed. `n_lines` (optional) is mixed in so a vocab
trained on a tiny probe and a production-scale vocab get distinct
fingerprints even when the token set happens to match.

This is the primary key used by `find_compatible_bundle` to decide
whether a saved model can be reused on a new corpus.
"""
function corpus_fingerprint(tokens::AbstractVector{<:AbstractString};
                            n_lines::Integer = 0)::String
    sorted = sort(unique(String[String(t) for t in tokens]))
    buf = IOBuffer()
    for t in sorted
        write(buf, t); write(buf, '\0')
    end
    write(buf, "|n="); write(buf, string(n_lines))
    return bytes2hex(sha256(take!(buf)))
end

"""
    BundleInfo(path, kind, schema_version, spec, metadata)

Lightweight view of a saved bundle returned by
[`list_bundles`] — avoids rehydrating the full payload when all
the caller wants is the header / spec / metadata.
"""
struct BundleInfo
    path::String
    kind::Symbol
    schema_version::UInt8
    spec::NamedTuple
    metadata::Dict{String, Any}
end

"""
    list_bundles(dir::AbstractString) -> Vector{BundleInfo}

Walk `dir` for `.jld2` files and peek at each one's header (`kind`,
`schema_version`, `spec`, `metadata`) without loading the full
payload. Silently skips non-bundle `.jld2` files so a mixed
directory can hold training state, checkpoints, etc.
"""
function list_bundles(dir::AbstractString)::Vector{BundleInfo}
    isdir(dir) || return BundleInfo[]
    out = BundleInfo[]
    for entry in sort(readdir(dir; join = true))
        endswith(entry, ".jld2") || continue
        isfile(entry) || continue
        info = try
            jldopen(entry, "r") do f
                BundleInfo(
                    entry,
                    Symbol(f["kind"]),
                    UInt8(f["schema_version"]),
                    f["spec"],
                    Dict{String, Any}(string(k) => v for (k, v) in f["metadata"]),
                )
            end
        catch
            nothing
        end
        info === nothing && continue
        push!(out, info)
    end
    return out
end

"""
    find_compatible_bundle(dir, kind::Symbol, fingerprint::AbstractString)
        -> Union{BundleInfo, Nothing}

Scan `dir` for a bundle of the requested `kind` whose
`metadata["corpus_fingerprint"]` matches `fingerprint`. Returns
the most recently modified match, or `nothing`. This is the
predicate behind `--reuse`: if it returns a bundle, the CLI
copies it to `--out` and skips training.
"""
function find_compatible_bundle(dir::AbstractString, kind::Symbol,
                                fingerprint::AbstractString
                                )::Union{BundleInfo, Nothing}
    infos = list_bundles(dir)
    candidates = filter(b ->
        b.kind == kind &&
        get(b.metadata, "corpus_fingerprint", "") == fingerprint,
        infos)
    isempty(candidates) && return nothing
    # Most-recent wins. `mtime` returns a Float64 (Unix secs).
    sort!(candidates; by = b -> mtime(b.path), rev = true)
    return first(candidates)
end

end # module Persistence
