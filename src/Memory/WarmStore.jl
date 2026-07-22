"""
    Memory.WarmStore

Short-term state that needs to survive a process restart and (when
backed by Redis) is shared across multiple `logcluster stream`
instances. Today the only consumer is the `novel_cluster` rule —
its `:seen` Set persists across boots so a `systemctl restart`
doesn't replay every novel-template alert from scratch — but the
trait is shaped to absorb sliding windows, reservoirs, and cooldown
keys as Rules.jl grows.

## Surface

- [`WarmStore`] — abstract parent.
- [`InProcWarmStore`] — no-op default; rules fall back to their
  in-process state.
- [`RedisWarmStore`] — connection + namespace + TTL; talks to
  [`RedisClient`].
- [`open_warm`] — parse a `--memory-warm` URL ("redis://…",
  "inproc", or `""`) into a store.
- [`namespace_for`] — `logcluster:<sha256(rules)[:12]>` so swapping
  the rules file gives a clean state.
- [`seen_member`] / [`add_member!`] / [`load_set`] — set operations
  used by `novel_cluster`.
"""
module WarmStoreModule

using ...Rules: Rules
using ..RedisClient: RedisClient, Client, connect_redis, sadd!, sismember,
                    smembers, ping
using SHA: sha256
using Sockets

export WarmStore, InProcWarmStore, RedisWarmStore,
       open_warm, namespace_for,
       seen_member, add_member!, load_set, healthy, flush_rule!

abstract type WarmStore end

"In-process / no-op warm store. Falls back to whatever per-rule state
the engine already keeps."
struct InProcWarmStore <: WarmStore end

"""
    RedisWarmStore(client, namespace; ttl_s = 0)

Redis-backed warm store. `namespace` prefixes every key so two
stream instances can share state safely. `ttl_s > 0` sets an
expiry on each key so a long-stopped stream doesn't accumulate
stale state forever (sets default to no TTL).
"""
mutable struct RedisWarmStore <: WarmStore
    client::Client
    namespace::String
    ttl_s::Int
end

"""
    open_warm(url::AbstractString; rules_fingerprint = "") -> WarmStore

Dispatch helper used by the CLI: empty / `"inproc"` returns an
[`InProcWarmStore`]; a `redis://…` URL connects and returns a
[`RedisWarmStore`] with the namespace derived from
`rules_fingerprint` (defaulting to a fresh sha256 so multiple
runtimes don't collide).
"""
function open_warm(url::AbstractString;
                   rules_fingerprint::AbstractString = "",
                   ttl_s::Integer = 0)
    s = String(url)
    if isempty(s) || s == "inproc"
        return InProcWarmStore()
    elseif startswith(s, "redis://")
        c = connect_redis(s)
        ns = "logcluster:" *
             (isempty(rules_fingerprint) ?
                bytes2hex(sha256(string(time())))[1:12] :
                rules_fingerprint[1:min(end, 12)])
        return RedisWarmStore(c, ns, Int(ttl_s))
    else
        throw(ArgumentError("--memory-warm must be empty, `inproc`, or " *
                            "redis://…; got `$s`"))
    end
end

function Base.close(s::RedisWarmStore)
    try; close(s.client); catch; end
    return nothing
end
Base.close(::InProcWarmStore) = nothing

"""
    namespace_for(rules_fingerprint) -> String

Build the Redis key namespace from a rules fingerprint. Exposed so
tests can verify two rule files don't collide.
"""
namespace_for(rules_fingerprint::AbstractString) =
    "logcluster:" * (isempty(rules_fingerprint) ? "default" :
                     rules_fingerprint[1:min(end, 12)])

"""
    healthy(s::WarmStore) -> Bool

Lightweight liveness check. Used by `doctor` and the boot path.
"""
healthy(::InProcWarmStore) = true
healthy(s::RedisWarmStore) = try; ping(s.client); catch; false; end

# ---------------------------------------------------------------------------
# Set operations — the novel_cluster rule's seen-id set lives here.
# ---------------------------------------------------------------------------

"""
    seen_member(s, rule_id, value) -> Bool

`true` when `value` has been recorded under `rule_id`. Always
`false` for [`InProcWarmStore`] — that path lives in rule state.
"""
seen_member(::InProcWarmStore, _rule_id, _value) = false

function seen_member(s::RedisWarmStore, rule_id::AbstractString,
                     value::AbstractString)
    return sismember(s.client, _set_key(s, rule_id), String(value))
end

"""
    add_member!(s, rule_id, value) -> Bool

Record `value` under `rule_id`. Returns `true` when the value was
newly added. No-op for [`InProcWarmStore`].
"""
add_member!(::InProcWarmStore, _rule_id, _value) = false

function add_member!(s::RedisWarmStore, rule_id::AbstractString,
                     value::AbstractString)
    return sadd!(s.client, _set_key(s, rule_id), [String(value)]) > 0
end

"""
    load_set(s, rule_id) -> Vector{String}

Hydrate every previously-seen value for `rule_id`. Used at stream
boot to repopulate per-rule state.
"""
load_set(::InProcWarmStore, _rule_id) = String[]

function load_set(s::RedisWarmStore, rule_id::AbstractString)
    return smembers(s.client, _set_key(s, rule_id))
end

@inline _set_key(s::RedisWarmStore, rule_id) =
    "$(s.namespace):set:$rule_id"

"""
    flush_rule!(s, rule_id)

Delete the stored set for `rule_id` (the `--memory-warm-flush-on-boot`
reset). No-op for [`InProcWarmStore`].
"""
flush_rule!(::InProcWarmStore, _rule_id) = nothing

function flush_rule!(s::RedisWarmStore, rule_id::AbstractString)
    RedisClient.del!(s.client, _set_key(s, rule_id))
    return nothing
end

# Plug WarmStore subtypes into the rules engine's warm-store hooks.
Rules._warm_load_set(s::WarmStore, rule_id) =
    load_set(s, String(rule_id))
Rules._warm_seen_member(s::WarmStore, rule_id, v) =
    seen_member(s, String(rule_id), String(v))
Rules._warm_add_member!(s::WarmStore, rule_id, v) =
    add_member!(s, String(rule_id), String(v))

end # module WarmStoreModule
