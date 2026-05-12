"""
    Rules

Declarative rule engine for [`Stream`]. Each rule is a small JSON
object with a `kind` field that selects a Julia evaluator; new rule
kinds are added by extending the dispatch in this file rather than by
parsing a custom expression DSL. Choice trades expressiveness for
auditability — the only thing operators can configure is the set of
pre-registered evaluators.

## Surface

- [`RuleSet`] — parsed rule bundle + per-rule mutable state + warmup
  bookkeeping.
- [`load_rules`] / [`default_rules`] — JSON loaders; the latter reads
  the bundled `src/rules/defaults.json`.
- [`evaluate`] — feed an `InferResult` dict to every rule, return the
  triggers that fired (cooldown-suppressed firings are silently
  dropped).
- [`snapshot`] — small `Dict{String, Any}` for the status heartbeat.

## Rule kinds

| `kind`            | What fires it                                                    |
|-------------------|------------------------------------------------------------------|
| `score_threshold` | `metric op value`; value may be a fixed number or `"auto:p99"` / `"auto:p995"` / `"auto:zscore:K"` (computed from the per-rule reservoir). |
| `novel_cluster`   | The model's `cluster_id` was never seen during warmup.           |
| `novel_token`     | The line contains a token not seen during warmup (rolling LRU).  |
| `rate_spike`      | Matched-line count in the rolling window exceeds `min_count` OR `baseline_multiplier · warmup_baseline`. |
| `volume_anomaly`  | Total lines in the rolling window exceed `baseline_multiplier · warmup_baseline`. |
| `keyword`         | Substring match on any of `keywords` against a configured field. |
| `regex`           | Regex match against a configured field.                          |

Each rule carries `severity ∈ {info, warn, crit}`, a `cooldown_s`
suppression window, and a `warmup_required` flag honoured by the
streaming loop.

## `InferResult` contract

The evaluator expects an `AbstractDict{String, Any}` with at least:

- `"line"`        — the (possibly masked) log line as a `String`.
- `"line_id"`     — monotonic 1-based id (`Int`).
- model namespaces e.g. `"drain"` => `Dict("cluster_id" => Int,
  "template" => String)`, `"transformer_decoder"` => `Dict("nll" =>
  Float64, ...)`.
- optional `"novelty"` => `Dict("value_novelty" => Float64)`.

Rule `metric` paths are dotted (`"transformer_decoder.nll"`) and
resolved by [`Rules._lookup`].
"""
module Rules

using JSON3
using Dates: Dates, DateTime, now, UTC
using Statistics: mean, std

export RuleSet, TriggerEvent, default_rules, load_rules, evaluate,
       snapshot, route_sinks

# ---------------------------------------------------------------------------
# Rule structs (one per kind for type-stable dispatch).
# ---------------------------------------------------------------------------

abstract type AbstractRule end

"""Rule id, severity, cooldown — shared by every kind. Stored as plain
fields on each concrete struct so dispatch stays cheap and there is no
abstract container indirection."""
struct ScoreThresholdRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    metric::String
    comparison::Symbol      # :gt | :ge | :lt | :le | :eq
    value_kind::Symbol      # :fixed | :p99 | :zscore
    value_arg::Float64      # fixed threshold, quantile, or z-score K
end

struct NovelClusterRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    model::String           # "drain" | "deep_kate" | …
end

struct NovelTokenRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    field::String           # default "line"
    lru_size::Int
end

struct RateSpikeRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    match_kind::Symbol            # :always | :regex | :keyword
    match_field::String
    match_regex::Union{Nothing, Regex}
    match_keywords::Vector{String}
    case_sensitive::Bool
    window_s::Float64
    min_count::Int
    baseline_multiplier::Float64
end

struct VolumeAnomalyRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    window_s::Float64
    baseline_multiplier::Float64
end

struct KeywordRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    field::String
    keywords::Vector{String}
    case_sensitive::Bool
end

struct RegexRule <: AbstractRule
    id::String
    severity::Symbol
    cooldown_s::Float64
    warmup_required::Bool
    field::String
    pattern::Regex
end

struct Route
    severities::Set{Symbol}
    rule_ids::Set{String}
    sinks::Vector{String}
end

struct SinkSpec
    name::String
    kind::Symbol            # :stdout | :webhook
    url::String
    method::String
    headers::Dict{String, String}
    secret_env::String
    format::Symbol          # :raw | :slack | :alertmanager
end

# ---------------------------------------------------------------------------
# State — held per-rule. Per-kind state shapes differ; we use a
# Dict{Symbol, Any} keyed by rule id to keep the RuleSet struct stable.
# ---------------------------------------------------------------------------

mutable struct WarmupState
    started_at::Float64
    lines_seen::Int
    n_required::Int
    seconds_required::Float64
    completed::Bool
end

"""
    RuleSet

In-memory rule bundle plus mutable per-rule state. The struct itself
is mostly immutable; only `state` dicts and `warmup` mutate on each
[`evaluate`] call.

`warm_store` is an optional hook into a long-term backing store
([`Memory.WarmStore`]). When set, the `novel_cluster` rule consults
it across evaluations so a process restart doesn't re-fire every
template the operator has already seen. Default is `nothing` — the
engine behaves exactly as before.
"""
mutable struct RuleSet
    version::Int
    defaults::Dict{Symbol, Any}
    rules::Vector{AbstractRule}
    routes::Vector{Route}
    sinks::Dict{String, SinkSpec}
    state::Dict{String, Dict{Symbol, Any}}
    warmup::WarmupState
    warm_store::Any   # ::Union{Nothing, Memory.WarmStore} — typed as Any to
                      # avoid a circular module dependency between Rules ↔ Memory.
end

"""
    TriggerEvent(rule_id, rule_kind, severity, line_id, line, fields, sinks, ts)

A single rule firing. The `fields` Dict carries kind-specific evidence
(matched keywords, threshold + observed value, cluster id, …); `sinks`
is the resolved list of sink names for the routing layer.
"""
struct TriggerEvent
    rule_id::String
    rule_kind::Symbol
    severity::Symbol
    line_id::Int
    line::String
    fields::Dict{String, Any}
    sinks::Vector{String}
    ts::DateTime
end

# ---------------------------------------------------------------------------
# Loading / parsing.
# ---------------------------------------------------------------------------

"""
    default_rules(; warmup_lines = 500, warmup_seconds = 30) -> RuleSet

Read the bundled rule set from `src/rules/defaults.json`.
"""
function default_rules(; warmup_lines::Integer = 500,
                       warmup_seconds::Real = 30.0)
    path = joinpath(@__DIR__, "rules", "defaults.json")
    return load_rules(path; warmup_lines = warmup_lines,
                      warmup_seconds = warmup_seconds)
end

"""
    load_rules(path::AbstractString; warmup_lines, warmup_seconds) -> RuleSet
    load_rules(io::IO;            warmup_lines, warmup_seconds) -> RuleSet

Parse a rules JSON file or `IO` stream into a [`RuleSet`]. Raises a
descriptive `ArgumentError` on schema problems (unknown `kind`,
duplicate rule ids, missing required fields, malformed `value`).
"""
function load_rules(path::AbstractString; warmup_lines::Integer = 500,
                    warmup_seconds::Real = 30.0)
    raw = JSON3.read(read(path, String))
    return _build_ruleset(raw, Int(warmup_lines), Float64(warmup_seconds))
end

function load_rules(io::IO; warmup_lines::Integer = 500,
                    warmup_seconds::Real = 30.0)
    raw = JSON3.read(read(io, String))
    return _build_ruleset(raw, Int(warmup_lines), Float64(warmup_seconds))
end

function _build_ruleset(raw, warmup_lines::Int, warmup_seconds::Float64)
    version = Int(get(raw, :version, 1))
    version == 1 || throw(ArgumentError(
        "unsupported rules schema version $version (expected 1)"))

    # Defaults block: severity / cooldown_s / warmup_required.
    d = Dict{Symbol, Any}()
    if haskey(raw, :defaults)
        for (k, v) in pairs(raw.defaults)
            d[Symbol(k)] = v
        end
    end

    # Rules.
    rules = AbstractRule[]
    if haskey(raw, :rules)
        for r in raw.rules
            push!(rules, _parse_rule(r, d))
        end
    end
    seen_ids = Set{String}()
    for r in rules
        r.id in seen_ids &&
            throw(ArgumentError("duplicate rule id: $(r.id)"))
        push!(seen_ids, r.id)
    end

    # Routes.
    routes = Route[]
    if haskey(raw, :routes)
        for rt in raw.routes
            push!(routes, _parse_route(rt))
        end
    end

    # Sinks.
    sinks = Dict{String, SinkSpec}()
    if haskey(raw, :sinks)
        for (name, spec) in pairs(raw.sinks)
            sinks[String(name)] = _parse_sink(String(name), spec)
        end
    end

    # Per-rule state.
    state = Dict{String, Dict{Symbol, Any}}()
    for r in rules
        state[r.id] = _initial_state(r)
    end

    warmup = WarmupState(time(), 0, warmup_lines, warmup_seconds, false)
    return RuleSet(version, d, rules, routes, sinks, state, warmup, nothing)
end

"""
    attach_warm_store!(rs::RuleSet, store) -> RuleSet

Bind a `Memory.WarmStore` to the rule set. Hydrates the in-process
`:seen` set on every `novel_cluster` rule from the store so a
restart doesn't replay alerts. Returns `rs` (chainable).
"""
function attach_warm_store!(rs::RuleSet, store)
    rs.warm_store = store
    for r in rs.rules
        r isa NovelClusterRule || continue
        # Cheap to call hydrator even when store == InProcWarmStore (no-op).
        previously = _warm_load_set(store, r.id)
        st = rs.state[r.id]
        st[:warm] = store
        for v in previously
            push!(st[:seen], v)
        end
    end
    return rs
end

# Warm-store hooks. The `::Nothing` methods cover the default
# (no warm store attached); `Memory.WarmStoreModule` adds methods
# on concrete `WarmStore` subtypes by extending these symbols
# directly.
_warm_load_set(::Nothing, _rule_id) = String[]
_warm_seen_member(::Nothing, _rule_id, _value) = false
_warm_add_member!(::Nothing, _rule_id, _value) = false

# Generic fallbacks for any non-`Nothing` store that hasn't yet had
# a method defined — keep the system safe to evaluate even before
# Memory loads.
_warm_load_set(_s, _rule_id) = String[]
_warm_seen_member(_s, _rule_id, _value) = false
_warm_add_member!(_s, _rule_id, _value) = false

# Shared helpers for parsing the common fields.
@inline function _common_fields(r, defaults::Dict{Symbol, Any})
    id = String(r.id)
    sev_raw = get(r, :severity, get(defaults, :severity, "warn"))
    sev = Symbol(sev_raw)
    sev in (:info, :warn, :crit) ||
        throw(ArgumentError("rule $id: invalid severity `$sev_raw`"))
    cd = Float64(get(r, :cooldown_s, get(defaults, :cooldown_s, 0.0)))
    wr = Bool(get(r, :warmup_required, get(defaults, :warmup_required, true)))
    return id, sev, cd, wr
end

function _parse_rule(r, defaults::Dict{Symbol, Any})
    kind = Symbol(r.kind)
    id, sev, cd, wr = _common_fields(r, defaults)

    if kind === :score_threshold
        metric = String(r.metric)
        comp_str = String(r.comparison)
        comp = comp_str == ">"  ? :gt :
               comp_str == ">=" ? :ge :
               comp_str == "<"  ? :lt :
               comp_str == "<=" ? :le :
               comp_str == "==" ? :eq :
               throw(ArgumentError("rule $id: invalid comparison `$comp_str`"))
        v = r.value
        if v isa Number
            return ScoreThresholdRule(id, sev, cd, wr, metric, comp, :fixed,
                                       Float64(v))
        else
            s = String(v)
            if s == "auto:p99"
                return ScoreThresholdRule(id, sev, cd, wr, metric, comp,
                                           :p99, 0.99)
            elseif s == "auto:p995"
                return ScoreThresholdRule(id, sev, cd, wr, metric, comp,
                                           :p99, 0.995)
            elseif startswith(s, "auto:zscore:")
                k = parse(Float64, s[length("auto:zscore:") + 1 : end])
                return ScoreThresholdRule(id, sev, cd, wr, metric, comp,
                                           :zscore, k)
            else
                throw(ArgumentError("rule $id: invalid value `$s`"))
            end
        end

    elseif kind === :novel_cluster
        model = String(get(r, :model, "drain"))
        return NovelClusterRule(id, sev, cd, wr, model)

    elseif kind === :novel_token
        field = String(get(r, :field, "line"))
        lru = Int(get(r, :lru_size, 8192))
        return NovelTokenRule(id, sev, cd, wr, field, lru)

    elseif kind === :rate_spike
        match_kind = :always
        match_field = "line"
        match_regex = nothing
        match_keywords = String[]
        case_sens = false
        if haskey(r, :match)
            m = r.match
            mk = Symbol(get(m, :kind, "regex"))
            match_kind = mk
            match_field = String(get(m, :field, "line"))
            if mk === :regex
                match_regex = Regex(String(m.pattern))
            elseif mk === :keyword
                match_keywords = [String(s) for s in m.keywords]
                case_sens = Bool(get(m, :case_sensitive, false))
            elseif mk !== :always
                throw(ArgumentError("rule $id: invalid match.kind `$mk`"))
            end
        end
        return RateSpikeRule(id, sev, cd, wr, match_kind, match_field,
                              match_regex, match_keywords, case_sens,
                              Float64(get(r, :window_s, 60.0)),
                              Int(get(r, :min_count, 0)),
                              Float64(get(r, :baseline_multiplier, 0.0)))

    elseif kind === :volume_anomaly
        return VolumeAnomalyRule(id, sev, cd, wr,
                                  Float64(get(r, :window_s, 60.0)),
                                  Float64(get(r, :baseline_multiplier, 5.0)))

    elseif kind === :keyword
        return KeywordRule(id, sev, cd, wr,
                           String(get(r, :field, "line")),
                           [String(s) for s in r.keywords],
                           Bool(get(r, :case_sensitive, false)))

    elseif kind === :regex
        return RegexRule(id, sev, cd, wr,
                         String(get(r, :field, "line")),
                         Regex(String(r.pattern)))

    else
        throw(ArgumentError("rule $id: unknown kind `$kind`"))
    end
end

function _parse_route(rt)
    sev_set = Set{Symbol}()
    rid_set = Set{String}()
    if haskey(rt, :match)
        m = rt.match
        if haskey(m, :severity)
            for s in m.severity
                push!(sev_set, Symbol(s))
            end
        end
        if haskey(m, :rule_id)
            for r in m.rule_id
                push!(rid_set, String(r))
            end
        end
    end
    sinks = String[String(s) for s in get(rt, :sinks, [])]
    return Route(sev_set, rid_set, sinks)
end

function _parse_sink(name::String, spec)
    url = String(get(spec, :url, ""))
    method = String(get(spec, :method, "POST"))
    headers = Dict{String, String}()
    if haskey(spec, :headers)
        for (k, v) in pairs(spec.headers)
            headers[String(k)] = String(v)
        end
    end
    secret_env = String(get(spec, :secret_env, ""))
    fmt_raw = get(spec, :format, "raw")
    fmt = Symbol(fmt_raw)
    fmt in (:raw, :slack, :alertmanager) ||
        throw(ArgumentError("sink $name: invalid format `$fmt_raw`"))

    kind = startswith(name, "stdout")  ? :stdout  :
           startswith(name, "webhook") ? :webhook :
           throw(ArgumentError(
               "sink $name: must start with `stdout` or `webhook:`"))
    return SinkSpec(name, kind, url, method, headers, secret_env, fmt)
end

# ---------------------------------------------------------------------------
# Per-rule state initialisation. Returns Dict{Symbol, Any} so per-kind
# evaluators can stash whatever they need (sets, sliding windows, EMAs).
# ---------------------------------------------------------------------------

const _RESERVOIR_MAX = 1024

_initial_state(::ScoreThresholdRule) =
    Dict{Symbol, Any}(:last_fired => -Inf, :reservoir => Float64[])

_initial_state(::NovelClusterRule) =
    Dict{Symbol, Any}(:last_fired => -Inf, :seen => Set{Any}())

_initial_state(::NovelTokenRule) =
    Dict{Symbol, Any}(:last_fired => -Inf,
                       :seen  => Set{String}(),
                       :order => String[])

_initial_state(::RateSpikeRule) =
    Dict{Symbol, Any}(:last_fired => -Inf,
                       :events => Float64[],
                       :baseline_count => 0,
                       :baseline_n => 0,
                       :baseline_window_count => 0.0)

_initial_state(::VolumeAnomalyRule) =
    Dict{Symbol, Any}(:last_fired => -Inf,
                       :events => Float64[],
                       :baseline_count => 0,
                       :baseline_n => 0,
                       :baseline_window_count => 0.0)

_initial_state(::KeywordRule) =
    Dict{Symbol, Any}(:last_fired => -Inf)

_initial_state(::RegexRule) =
    Dict{Symbol, Any}(:last_fired => -Inf)

# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

"Resolve a dotted path against a JSON-like dict tree. Returns
`nothing` on missing keys or non-dict intermediates."
function _lookup(ir, path::AbstractString)
    v = ir
    for p in split(path, '.')
        v isa AbstractDict || return nothing
        # Look up under both String and Symbol — JSON3 reads symbol keys.
        if haskey(v, String(p))
            v = v[String(p)]
        elseif haskey(v, Symbol(p))
            v = v[Symbol(p)]
        else
            return nothing
        end
    end
    return v
end

@inline function _get_field(ir, field::AbstractString)
    return _lookup(ir, field)
end

function _bump_warmup!(rs::RuleSet)
    rs.warmup.completed && return true
    rs.warmup.lines_seen += 1
    elapsed = time() - rs.warmup.started_at
    # "Buffer the first N lines" — strict `>` so line N is still warmup
    # and line N+1 is the first post-warmup line. With N = 0, the very
    # first line already satisfies `1 > 0` and rules fire immediately.
    done = rs.warmup.lines_seen > rs.warmup.n_required ||
           elapsed >= rs.warmup.seconds_required
    if done && !rs.warmup.completed
        rs.warmup.completed = true
        _freeze_baselines!(rs)
    end
    return done
end

# Once warmup completes, normalise rate / volume baselines to a
# per-window expectation: `baseline_window_count =
# (baseline_count / max(1, elapsed)) * window_s`.
function _freeze_baselines!(rs::RuleSet)
    elapsed = max(time() - rs.warmup.started_at, 1e-6)
    for r in rs.rules
        st = rs.state[r.id]
        if r isa RateSpikeRule
            rate = Float64(st[:baseline_count]) / elapsed
            st[:baseline_window_count] = rate * r.window_s
        elseif r isa VolumeAnomalyRule
            rate = Float64(st[:baseline_count]) / elapsed
            st[:baseline_window_count] = rate * r.window_s
        end
    end
end

"""
    evaluate(rs::RuleSet, ir::AbstractDict) -> Vector{TriggerEvent}

Apply every rule in `rs` to one `InferResult` dict. Returns the
triggers that fired this line (cooldown-suppressed firings are
silently dropped). Mutates per-rule state on `rs.state`.
"""
function evaluate(rs::RuleSet, ir::AbstractDict)
    triggers = TriggerEvent[]
    now_t = time()
    warmup_done = _bump_warmup!(rs)
    line_id = Int(get(ir, "line_id", 0))
    line    = String(get(ir, "line", ""))

    for rule in rs.rules
        st = rs.state[rule.id]
        if rule.warmup_required && !warmup_done
            _ingest!(rule, ir, st, now_t)
            continue
        end
        if (now_t - Float64(st[:last_fired])) < rule.cooldown_s
            # Still let stateful rules ingest data even when cooldown-suppressed
            # so they don't lose their windows during a quiet period.
            _ingest!(rule, ir, st, now_t)
            continue
        end
        fired, fields = _fire_check(rule, ir, st, now_t)
        if fired
            st[:last_fired] = now_t
            push!(triggers, TriggerEvent(
                rule.id, _kind_symbol(rule), rule.severity,
                line_id, line, fields, route_sinks(rs, rule),
                Dates.now(UTC)))
        end
    end
    return triggers
end

# ---------------------------------------------------------------------------
# Warmup ingestion — accumulate baselines without firing.
# ---------------------------------------------------------------------------

function _ingest!(r::ScoreThresholdRule, ir, st, _t)
    v = _lookup(ir, r.metric)
    if v isa Number
        push!(st[:reservoir], Float64(v))
        if length(st[:reservoir]) > _RESERVOIR_MAX
            popfirst!(st[:reservoir])
        end
    end
end

function _ingest!(r::NovelClusterRule, ir, st, _t)
    cid = _lookup(ir, "$(r.model).cluster_id")
    cid === nothing && return nothing
    cid_s = string(cid)
    push!(st[:seen], cid_s)
    warm = get(st, :warm, nothing)
    if warm !== nothing
        try
            _warm_add_member!(warm, r.id, cid_s)
        catch
        end
    end
    return nothing
end

function _ingest!(r::NovelTokenRule, ir, st, _t)
    s = _get_field(ir, r.field)
    s isa AbstractString || return nothing
    for tok in split(s)
        s2 = String(tok)
        if !(s2 in st[:seen])
            push!(st[:seen], s2)
            push!(st[:order], s2)
            if length(st[:seen]) > r.lru_size
                old = popfirst!(st[:order])
                delete!(st[:seen], old)
            end
        end
    end
    return nothing
end

function _ingest!(r::RateSpikeRule, ir, st, t)
    _matches_rate(r, ir) && (st[:baseline_count] += 1)
    st[:baseline_n] += 1
    # Keep the window populated so the firing check has real data the
    # moment warmup ends.
    if _matches_rate(r, ir)
        push!(st[:events], t)
    end
    _trim_window!(st[:events], t, r.window_s)
end

function _ingest!(r::VolumeAnomalyRule, ir, st, t)
    st[:baseline_count] += 1
    st[:baseline_n] += 1
    push!(st[:events], t)
    _trim_window!(st[:events], t, r.window_s)
end

_ingest!(::KeywordRule, _ir, _st, _t) = nothing
_ingest!(::RegexRule,   _ir, _st, _t) = nothing

# ---------------------------------------------------------------------------
# Fire check, per kind.
# Each `_fire_check` returns `(fired::Bool, fields::Dict{String, Any})`.
# ---------------------------------------------------------------------------

function _fire_check(r::ScoreThresholdRule, ir, st, _t)
    v = _lookup(ir, r.metric)
    v isa Number || return (false, Dict{String, Any}())
    fv = Float64(v)

    threshold = if r.value_kind === :fixed
        r.value_arg
    elseif r.value_kind === :p99
        _quantile(st[:reservoir], r.value_arg)
    elseif r.value_kind === :zscore
        if length(st[:reservoir]) >= 2
            mean(st[:reservoir]) + r.value_arg * std(st[:reservoir])
        else
            Inf
        end
    else
        Inf
    end

    fired = _compare(r.comparison, fv, threshold)

    # Update the reservoir live — `auto:*` thresholds keep adapting.
    push!(st[:reservoir], fv)
    if length(st[:reservoir]) > _RESERVOIR_MAX
        popfirst!(st[:reservoir])
    end

    fields = Dict{String, Any}(
        "metric"    => r.metric,
        "value"     => fv,
        "threshold" => threshold,
    )
    return (fired, fields)
end

function _fire_check(r::NovelClusterRule, ir, st, _t)
    cid = _lookup(ir, "$(r.model).cluster_id")
    cid === nothing && return (false, Dict{String, Any}())
    # Canonicalise the cluster id as String so the warm store and the
    # in-memory cache speak the same language across restarts.
    cid_s = string(cid)
    novel = !(cid_s in st[:seen])
    push!(st[:seen], cid_s)
    warm = get(st, :warm, nothing)
    if warm !== nothing && novel
        try
            _warm_add_member!(warm, r.id, cid_s)
        catch e
            @debug "warm_store add_member! failed" rule=r.id exception=e
        end
    end
    return (novel, Dict{String, Any}(
        "model"      => r.model,
        "cluster_id" => cid,
    ))
end

function _fire_check(r::NovelTokenRule, ir, st, _t)
    s = _get_field(ir, r.field)
    s isa AbstractString || return (false, Dict{String, Any}())
    novel = String[]
    for tok in split(s)
        s2 = String(tok)
        if !(s2 in st[:seen])
            push!(novel, s2)
            push!(st[:seen], s2)
            push!(st[:order], s2)
            if length(st[:seen]) > r.lru_size
                old = popfirst!(st[:order])
                delete!(st[:seen], old)
            end
        end
    end
    return (!isempty(novel), Dict{String, Any}("novel_tokens" => novel))
end

function _fire_check(r::RateSpikeRule, ir, st, t)
    _matches_rate(r, ir) && push!(st[:events], t)
    _trim_window!(st[:events], t, r.window_s)
    count = length(st[:events])
    baseline = Float64(st[:baseline_window_count])

    fired_min = r.min_count > 0 && count >= r.min_count
    fired_mult = r.baseline_multiplier > 0 && baseline > 0 &&
                  count >= r.baseline_multiplier * baseline

    fields = Dict{String, Any}(
        "count_in_window" => count,
        "window_s"        => r.window_s,
        "baseline"        => baseline,
        "min_count"       => r.min_count,
    )
    return ((fired_min || fired_mult), fields)
end

function _fire_check(r::VolumeAnomalyRule, ir, st, t)
    push!(st[:events], t)
    _trim_window!(st[:events], t, r.window_s)
    count = length(st[:events])
    baseline = Float64(st[:baseline_window_count])
    fired = baseline > 0 && count >= r.baseline_multiplier * baseline
    fields = Dict{String, Any}(
        "count_in_window" => count,
        "window_s"        => r.window_s,
        "baseline"        => baseline,
    )
    return (fired, fields)
end

function _fire_check(r::KeywordRule, ir, _st, _t)
    s = _get_field(ir, r.field)
    s isa AbstractString || return (false, Dict{String, Any}())
    haystack = r.case_sensitive ? s : lowercase(s)
    matched = String[]
    for kw in r.keywords
        needle = r.case_sensitive ? kw : lowercase(kw)
        occursin(needle, haystack) && push!(matched, kw)
    end
    return (!isempty(matched), Dict{String, Any}("matched" => matched))
end

function _fire_check(r::RegexRule, ir, _st, _t)
    s = _get_field(ir, r.field)
    s isa AbstractString || return (false, Dict{String, Any}())
    m = match(r.pattern, s)
    m === nothing && return (false, Dict{String, Any}())
    return (true, Dict{String, Any}("match" => String(m.match)))
end

# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

function _matches_rate(r::RateSpikeRule, ir)
    r.match_kind === :always && return true
    s = _get_field(ir, r.match_field)
    s isa AbstractString || return false
    if r.match_kind === :regex
        return r.match_regex === nothing ? false : occursin(r.match_regex, s)
    elseif r.match_kind === :keyword
        haystack = r.case_sensitive ? s : lowercase(s)
        for kw in r.match_keywords
            needle = r.case_sensitive ? kw : lowercase(kw)
            occursin(needle, haystack) && return true
        end
        return false
    end
    return false
end

function _trim_window!(events::Vector{Float64}, now_t::Float64, window_s::Float64)
    cutoff = now_t - window_s
    while !isempty(events) && events[1] < cutoff
        popfirst!(events)
    end
    return events
end

function _quantile(v::Vector{Float64}, q::Real)
    isempty(v) && return Inf
    s = sort(v)
    idx = clamp(ceil(Int, q * length(s)), 1, length(s))
    return s[idx]
end

@inline function _compare(op::Symbol, x::Float64, y::Float64)
    op === :gt && return x > y
    op === :ge && return x >= y
    op === :lt && return x < y
    op === :le && return x <= y
    op === :eq && return x == y
    return false
end

_kind_symbol(::ScoreThresholdRule) = :score_threshold
_kind_symbol(::NovelClusterRule)   = :novel_cluster
_kind_symbol(::NovelTokenRule)     = :novel_token
_kind_symbol(::RateSpikeRule)      = :rate_spike
_kind_symbol(::VolumeAnomalyRule)  = :volume_anomaly
_kind_symbol(::KeywordRule)        = :keyword
_kind_symbol(::RegexRule)          = :regex

"""
    route_sinks(rs::RuleSet, rule) -> Vector{String}

Resolve which sink names a triggered rule routes to. `"stdout"` is
always included; explicit routes can add webhook sinks.
"""
function route_sinks(rs::RuleSet, rule::AbstractRule)
    sinks = String["stdout"]
    for rt in rs.routes
        sev_ok = isempty(rt.severities) || rule.severity in rt.severities
        rid_ok = isempty(rt.rule_ids)   || rule.id      in rt.rule_ids
        sev_ok && rid_ok && append!(sinks, rt.sinks)
    end
    return unique!(sinks)
end

"""
    snapshot(rs::RuleSet) -> Dict{String, Any}

Lightweight status payload for the `event:"status"` heartbeat.
"""
function snapshot(rs::RuleSet)
    return Dict{String, Any}(
        "warmup" => Dict{String, Any}(
            "completed"        => rs.warmup.completed,
            "lines_seen"       => rs.warmup.lines_seen,
            "n_required"       => rs.warmup.n_required,
            "seconds_required" => rs.warmup.seconds_required,
        ),
        "rules"   => [r.id for r in rs.rules],
        "n_rules" => length(rs.rules),
    )
end

end # module Rules
