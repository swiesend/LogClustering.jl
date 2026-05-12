"""
    Memory.PatternCatalog

User-curated patterns persisted in the SQLite store. Each pattern
is a tagged matcher (drain template id, regex, or keyword set) with
curator metadata: name, description, severity, cooldown. Operators
pin patterns from interesting triggers and (optionally) auto-promote
them to rules on subsequent `stream` runs.

## Surface

- [`Pattern`] — hydrated row type returned by [`list`].
- [`pin_from_trigger`] / [`pin_manual`] — write entries.
- [`list`] / [`get_pattern`] — read entries.
- [`enable!`] / [`delete!`] — mutate enabled-flag.
- [`match_line`] — evaluate every enabled pattern against a single
  `InferResult` dict; returns the ids of matching patterns plus
  per-pattern evidence.
- [`to_rules`] — synthesise `Rules.AbstractRule` instances from
  enabled patterns so they slot into the existing rule engine.
"""
module PatternCatalog

using ..SQLiteStore: SQLiteStore
using ..SQLiteStore: _rows, _exec!
using ...Rules: Rules
using JSON3
using Dates: Dates, DateTime, now, UTC
using SQLite: SQLite, DB

export Pattern, pin_from_trigger, pin_manual, list, get_pattern,
       enable!, delete!, match_line, to_rules

# ---------------------------------------------------------------------------
# Types.
# ---------------------------------------------------------------------------

"""
    Pattern

Hydrated pattern row. `match_kind ∈ (:drain, :regex, :keyword)`
selects which `match_*` field drives matching.
"""
struct Pattern
    id::Int
    name::String
    description::String
    severity::Symbol
    cooldown_s::Float64
    match_kind::Symbol
    match_drain_template_id::Union{Nothing, Int}
    match_regex::Union{Nothing, Regex}
    match_keywords::Vector{String}
    case_sensitive::Bool
    created_by::String
    created_at::String
    enabled::Bool
end

_iso(dt::DateTime) = string(Dates.format(dt,
                            Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

# ---------------------------------------------------------------------------
# Pinning helpers.
# ---------------------------------------------------------------------------

"""
    pin_from_trigger(db, trigger_id; name, description = "",
                     severity = :warn, match_kind = :drain,
                     cooldown_s = 60, created_by = "") -> Int

Pin a pattern based on an existing trigger row. The `match_kind`
choice decides the column populated:

  - `:drain`   — use the trigger's `drain_cluster_id` as the
    template id (fails when null).
  - `:keyword` — use the words in the trigger's `line` as a keyword
    set (one-shot — operators usually edit the pattern after).
  - `:regex`   — escape the line into a literal regex match (again,
    operators usually narrow it).

Returns the new pattern id.
"""
function pin_from_trigger(db::DB, trigger_id::Integer;
                          name::AbstractString,
                          description::AbstractString = "",
                          severity::Symbol = :warn,
                          match_kind::Symbol = :drain,
                          cooldown_s::Real = 60.0,
                          created_by::AbstractString = "",
                          case_sensitive::Bool = false)
    trow = _rows(db,
        "SELECT line, drain_cluster_id FROM triggers WHERE id = ?",
        (Int(trigger_id),))
    isempty(trow) && throw(ArgumentError("trigger $trigger_id not found"))
    line = String(trow[1].line)
    dcid = trow[1].drain_cluster_id
    if match_kind === :drain
        dcid isa Missing && throw(ArgumentError(
            "trigger $trigger_id has no drain_cluster_id; pick another match_kind"))
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :drain,
            match_drain_template_id = Int(dcid),
            created_by = created_by,
            case_sensitive = case_sensitive)
    elseif match_kind === :keyword
        # Take up to 5 alpha-num words from the line as the keyword set.
        toks = filter(t -> !isempty(t),
                      split(replace(line, r"[^A-Za-z0-9]+" => " ")))
        keep = String[String(t) for t in toks][1:min(end, 5)]
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :keyword,
            match_keywords = keep,
            case_sensitive = case_sensitive,
            created_by = created_by)
    elseif match_kind === :regex
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :regex,
            match_regex = _escape_regex(line),
            case_sensitive = case_sensitive,
            created_by = created_by)
    else
        throw(ArgumentError("match_kind must be :drain | :keyword | :regex"))
    end
end

"""
    pin_manual(db; name, match_kind, [...other fields...]) -> Int

Author a pattern from scratch. `match_kind` is mandatory and selects
which of `match_drain_template_id` / `match_regex` / `match_keywords`
must be non-empty.
"""
function pin_manual(db::DB;
                    name::AbstractString,
                    description::AbstractString = "",
                    severity::Symbol = :warn,
                    cooldown_s::Real = 60.0,
                    match_kind::Symbol,
                    match_drain_template_id::Union{Nothing, Integer} = nothing,
                    match_regex::AbstractString = "",
                    match_keywords::Vector{<:AbstractString} = String[],
                    case_sensitive::Bool = false,
                    created_by::AbstractString = "")
    if match_kind === :drain
        match_drain_template_id === nothing && throw(ArgumentError(
            "match_kind = :drain needs match_drain_template_id"))
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :drain,
            match_drain_template_id = Int(match_drain_template_id),
            created_by = created_by, case_sensitive = case_sensitive)
    elseif match_kind === :regex
        isempty(match_regex) && throw(ArgumentError(
            "match_kind = :regex needs match_regex"))
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :regex,
            match_regex = String(match_regex),
            created_by = created_by, case_sensitive = case_sensitive)
    elseif match_kind === :keyword
        isempty(match_keywords) && throw(ArgumentError(
            "match_kind = :keyword needs match_keywords"))
        return _insert_pattern!(db; name = name, description = description,
            severity = severity, cooldown_s = cooldown_s,
            match_kind = :keyword,
            match_keywords = String[String(k) for k in match_keywords],
            created_by = created_by, case_sensitive = case_sensitive)
    else
        throw(ArgumentError("match_kind must be :drain | :keyword | :regex"))
    end
end

function _insert_pattern!(db::DB;
                          name::AbstractString,
                          description::AbstractString = "",
                          severity::Symbol = :warn,
                          cooldown_s::Real = 60.0,
                          match_kind::Symbol,
                          match_drain_template_id = nothing,
                          match_regex::AbstractString = "",
                          match_keywords::Vector{<:AbstractString} = String[],
                          case_sensitive::Bool = false,
                          created_by::AbstractString = "")
    _exec!(db,
        "INSERT INTO patterns (name, description, severity, cooldown_s, " *
        "match_kind, match_drain_template_id, match_regex, " *
        "match_keywords_json, case_sensitive, created_by, created_at, " *
        "enabled) " *
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (String(name), String(description), String(severity),
         Float64(cooldown_s), String(match_kind),
         match_drain_template_id === nothing ? nothing : Int(match_drain_template_id),
         isempty(match_regex) ? nothing : String(match_regex),
         JSON3.write(match_keywords),
         case_sensitive ? 1 : 0,
         String(created_by), _iso(now(UTC))))
    return Int(SQLite.last_insert_rowid(db))
end

# ---------------------------------------------------------------------------
# Read / mutate.
# ---------------------------------------------------------------------------

"""
    list(db; enabled_only::Bool = true) -> Vector{Pattern}
"""
function list(db::DB; enabled_only::Bool = true)
    sql = "SELECT id, name, description, severity, cooldown_s, " *
          "match_kind, match_drain_template_id, match_regex, " *
          "match_keywords_json, case_sensitive, created_by, created_at, " *
          "enabled FROM patterns"
    if enabled_only
        sql *= " WHERE enabled = 1"
    end
    sql *= " ORDER BY id ASC"
    rows = _rows(db, sql, ())
    return Pattern[_hydrate(r) for r in rows]
end

"""
    get_pattern(db, pattern_id) -> Union{Pattern, Nothing}
"""
function get_pattern(db::DB, pattern_id::Integer)
    rows = _rows(db,
        "SELECT id, name, description, severity, cooldown_s, " *
        "match_kind, match_drain_template_id, match_regex, " *
        "match_keywords_json, case_sensitive, created_by, created_at, " *
        "enabled FROM patterns WHERE id = ?",
        (Int(pattern_id),))
    isempty(rows) && return nothing
    return _hydrate(rows[1])
end

"""
    enable!(db, pattern_id, on::Bool)
"""
function enable!(db::DB, pattern_id::Integer, on::Bool)
    _exec!(db, "UPDATE patterns SET enabled = ? WHERE id = ?",
           (on ? 1 : 0, Int(pattern_id)))
    return nothing
end

"""
    delete!(db, pattern_id)
"""
function delete!(db::DB, pattern_id::Integer)
    _exec!(db, "DELETE FROM patterns WHERE id = ?", (Int(pattern_id),))
    return nothing
end

function _hydrate(r)
    kws = r.match_keywords_json isa Missing ? String[] :
          [String(k) for k in JSON3.read(String(r.match_keywords_json))]
    return Pattern(
        Int(r.id),
        String(r.name),
        r.description isa Missing ? "" : String(r.description),
        Symbol(r.severity),
        Float64(r.cooldown_s),
        Symbol(r.match_kind),
        r.match_drain_template_id isa Missing ? nothing : Int(r.match_drain_template_id),
        r.match_regex isa Missing || r.match_regex === nothing ? nothing :
            Regex(String(r.match_regex)),
        kws,
        (r.case_sensitive isa Missing ? 0 : Int(r.case_sensitive)) != 0,
        r.created_by isa Missing ? "" : String(r.created_by),
        String(r.created_at),
        Int(r.enabled) != 0,
    )
end

# ---------------------------------------------------------------------------
# Online matching + rule synthesis.
# ---------------------------------------------------------------------------

"""
    match_line(p::Pattern, ir::AbstractDict) -> Bool

Test one pattern against an InferResult dict. Drain matches when
`drain.cluster_id` equals the pinned template id; regex / keyword
matches operate on `ir["line"]`.
"""
function match_line(p::Pattern, ir::AbstractDict)
    if p.match_kind === :drain
        drain = get(ir, "drain", nothing)
        drain isa AbstractDict || return false
        cid = get(drain, "cluster_id", nothing)
        cid === nothing && return false
        return Int(cid) == Int(p.match_drain_template_id)
    elseif p.match_kind === :regex
        p.match_regex === nothing && return false
        line = String(get(ir, "line", ""))
        return occursin(p.match_regex, line)
    elseif p.match_kind === :keyword
        line = String(get(ir, "line", ""))
        hay = p.case_sensitive ? line : lowercase(line)
        for kw in p.match_keywords
            needle = p.case_sensitive ? kw : lowercase(kw)
            occursin(needle, hay) && return true
        end
        return false
    end
    return false
end

"""
    to_rules(patterns) -> Vector{Rules.AbstractRule}

Synthesise a `Rules.AbstractRule` for each enabled keyword / regex
pattern so the existing rule engine fires on them with no further
glue. Drain-template patterns are NOT synthesised — they get an
exact `cluster_id` match via a sidecar pass in `cmd_stream`
(see [`match_line`]). Rule ids are `"pattern:<id>:<name>"` so
trigger records flag the origin.
"""
function to_rules(patterns::AbstractVector{Pattern})
    out = Rules.AbstractRule[]
    for p in patterns
        p.enabled || continue
        rid = "pattern:$(p.id):$(p.name)"
        if p.match_kind === :keyword
            push!(out, Rules.KeywordRule(rid, p.severity, p.cooldown_s,
                false, "line", p.match_keywords, p.case_sensitive))
        elseif p.match_kind === :regex
            p.match_regex === nothing && continue
            push!(out, Rules.RegexRule(rid, p.severity, p.cooldown_s,
                false, "line", p.match_regex))
        end
        # :drain kinds are checked outside the rule engine — see
        # `PatternCatalog.match_line` and cmd_stream's drain-match
        # sidecar.
    end
    return out
end

"""
    drain_patterns(patterns) -> Vector{Pattern}

Filter enabled patterns whose match_kind is :drain. cmd_stream
runs these via [`match_line`] on every line and emits a synthetic
TriggerEvent for each match.
"""
drain_patterns(patterns::AbstractVector{Pattern}) =
    Pattern[p for p in patterns if p.enabled && p.match_kind === :drain]

# Pure-strings helper for "literal regex" pinning.
function _escape_regex(s::AbstractString)
    # Escape Regex metacharacters; the result is a literal-match
    # regex bracketed by line anchors so it doesn't blow up on
    # adjacent text.
    escaped = replace(String(s), r"([.\\^$*+?()[\]{}|])" => s"\\\1")
    return "^" * escaped * "\$"
end

end # module PatternCatalog
