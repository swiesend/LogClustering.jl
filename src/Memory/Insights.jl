"""
    Memory.Insights

Read-only analytics over the SQLite memory store. Each function
takes an open `DB` plus a time window and returns a `Vector` of
small NamedTuples — easy to render as TSV / JSON / Markdown without
new dependencies.

The temporal-pattern view (`episodes`) reuses
`Mining.Episodes.mv_span` against a cluster-id sequence pulled from
the store, so it composes cleanly with the existing batch RCA
pipeline.

## Surface

- [`top_rules_window`] — rule_id counts grouped by severity.
- [`novel_clusters_window`] — Drain clusters first seen in the window.
- [`burstiness`] — bucketed counts for one rule.
- [`cluster_view`] — bucketed counts for one drain cluster.
- [`transitions`] — top P(next | this) cluster transitions.
- [`episodes`] — `mv_span` over the cluster-id sequence.
- [`pinned_summary`] — pattern_matches grouped by pattern.
"""
module Insights

using ..SQLiteStore: SQLiteStore, epoch_ms_since, _rows, lines_with_embeddings
using ...Episodes: Episodes, mv_span
using ...Pipeline: umap_reduce
using SQLite: DB

export top_rules_window, novel_clusters_window, burstiness,
       cluster_view, transitions, episodes, pinned_summary,
       embedding_scatter, trigger_timeline

# ---------------------------------------------------------------------------
# Top rules.
# ---------------------------------------------------------------------------

"""
    top_rules_window(db; since, until = nothing, limit = 20) -> Vector{NT}

Per-rule trigger counts within `[since, until]`. Each row carries
`(rule_id, severity, n, last_seen_ms)`.
"""
function top_rules_window(db::DB;
                          since::Integer,
                          until::Union{Nothing, Integer} = nothing,
                          limit::Integer = 20)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    return _rows(db,
        "SELECT rule_id, severity, COUNT(*) AS n, " *
        "MAX(ts_epoch_ms) AS last_seen_ms " *
        "FROM triggers " *
        "WHERE ts_epoch_ms >= ? AND ts_epoch_ms <= ? " *
        "GROUP BY rule_id, severity " *
        "ORDER BY n DESC LIMIT ?",
        (Int(since), until_ms, Int(limit)))
end

# ---------------------------------------------------------------------------
# Novel clusters.
# ---------------------------------------------------------------------------

"""
    novel_clusters_window(db; since, until = nothing) -> Vector{NT}

Drain `cluster_id`s whose first sighting in the database falls
within `[since, until]`.
"""
function novel_clusters_window(db::DB;
                                since::Integer,
                                until::Union{Nothing, Integer} = nothing)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    return _rows(db, """
        SELECT cluster_id, first_seen_ms FROM (
            SELECT drain_cluster_id AS cluster_id,
                   MIN(ts_epoch_ms)  AS first_seen_ms
            FROM triggers
            WHERE drain_cluster_id IS NOT NULL
            GROUP BY drain_cluster_id
        )
        WHERE first_seen_ms >= ? AND first_seen_ms <= ?
        ORDER BY first_seen_ms ASC
        """,
        (Int(since), until_ms))
end

# ---------------------------------------------------------------------------
# Burstiness for a single rule.
# ---------------------------------------------------------------------------

"""
    burstiness(db; rule_id, since, until = nothing, bucket_s = 60) -> Vector{NT}

Bucketed trigger counts for one rule. Each row: `(bucket_start_ms, n)`.
"""
function burstiness(db::DB;
                    rule_id::AbstractString,
                    since::Integer,
                    until::Union{Nothing, Integer} = nothing,
                    bucket_s::Real = 60)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    bucket_ms = Int(round(bucket_s * 1000))
    return _rows(db, """
        SELECT (ts_epoch_ms / ?) * ? AS bucket_start_ms, COUNT(*) AS n
        FROM triggers
        WHERE rule_id = ? AND ts_epoch_ms >= ? AND ts_epoch_ms <= ?
        GROUP BY bucket_start_ms
        ORDER BY bucket_start_ms ASC
        """,
        (bucket_ms, bucket_ms, String(rule_id), Int(since), until_ms))
end

# ---------------------------------------------------------------------------
# Cluster timeline (alias for backwards compat with the SQLiteStore helper).
# ---------------------------------------------------------------------------

cluster_view(db::DB; cluster_id::Integer,
             since::Integer,
             until::Union{Nothing, Integer} = nothing,
             bucket_s::Real = 60) =
    SQLiteStore.cluster_timeline(db; cluster_id = cluster_id,
        since = since, until = until, bucket_s = bucket_s)

# ---------------------------------------------------------------------------
# Cluster transitions — bigram counts over the time-ordered cluster sequence.
# ---------------------------------------------------------------------------

"""
    transitions(db; since, until = nothing, top = 20) -> Vector{NT}

Top-K most frequent (this -> next) drain cluster id transitions in
the window. Lines / triggers with no `drain_cluster_id` are skipped.
"""
function transitions(db::DB;
                     since::Integer,
                     until::Union{Nothing, Integer} = nothing,
                     top::Integer = 20)
    seq = SQLiteStore.cluster_id_sequence(db; since = since, until = until)
    length(seq) < 2 && return NamedTuple[]
    counts = Dict{Tuple{Int, Int}, Int}()
    @inbounds for i in 1:length(seq) - 1
        k = (seq[i], seq[i + 1])
        counts[k] = get(counts, k, 0) + 1
    end
    pairs = collect(counts)
    sort!(pairs; by = p -> -p.second)
    out = NamedTuple[]
    for (i, ((a, b), n)) in enumerate(pairs)
        i > top && break
        push!(out, (from = a, to = b, n = n))
    end
    return out
end

# ---------------------------------------------------------------------------
# Episode mining.
# ---------------------------------------------------------------------------

"""
    episodes(db; since, until = nothing, min_sup = 3, max_gap = 20,
             max_time_duration = 50, max_repetitions = 0,
             top = 20) -> Vector{NT}

Top-K episodes mined via `Mining.Episodes.mv_span` over the
time-ordered drain-cluster sequence from the window. Each row:
`(pattern, support)` where `pattern` is a Vector{Int}.
"""
function episodes(db::DB;
                  since::Integer,
                  until::Union{Nothing, Integer} = nothing,
                  min_sup::Integer = 3,
                  max_gap::Integer = 20,
                  max_time_duration::Integer = 50,
                  max_repetitions::Integer = 0,
                  top::Integer = 20)
    seq = SQLiteStore.cluster_id_sequence(db; since = since, until = until)
    length(seq) < min_sup && return NamedTuple[]
    occs = mv_span(seq;
        min_sup = Int(min_sup),
        max_gap = Int(max_gap),
        max_time_duration = Int(max_time_duration),
        max_repetitions = Int(max_repetitions),
    )
    pairs = collect(occs)
    sort!(pairs; by = p -> -length(p.second))
    out = NamedTuple[]
    for (i, (pat, occurrences)) in enumerate(pairs)
        i > top && break
        push!(out, (pattern = Vector{Int}(pat), support = length(occurrences)))
    end
    return out
end

# ---------------------------------------------------------------------------
# Pinned-pattern summary.
# ---------------------------------------------------------------------------

"""
    pinned_summary(db; since, until = nothing) -> Vector{NT}

Per-pattern match counts inside the window, joined with the
`patterns` table so names + severities are right there in the row.
"""
function pinned_summary(db::DB;
                        since::Integer,
                        until::Union{Nothing, Integer} = nothing)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    return _rows(db, """
        SELECT p.id AS pattern_id, p.name AS name, p.severity AS severity,
               COUNT(pm.id) AS n,
               MAX(pm.ts_epoch_ms) AS last_seen_ms
        FROM patterns p
        LEFT JOIN pattern_matches pm
          ON pm.pattern_id = p.id
         AND pm.ts_epoch_ms >= ? AND pm.ts_epoch_ms <= ?
        WHERE p.enabled = 1
        GROUP BY p.id
        ORDER BY n DESC
        """,
        (Int(since), until_ms))
end

# ---------------------------------------------------------------------------
# Overall trigger timeline (all rules), bucketed.
# ---------------------------------------------------------------------------

"""
    trigger_timeline(db; since, until = nothing, bucket_s = 3600) -> Vector{NT}

Bucketed count of *all* triggers over `[since, until]`. Each row:
`(bucket_start_ms, n)`. Feeds the HTML report's timeline bar chart.
"""
function trigger_timeline(db::DB;
                          since::Integer,
                          until::Union{Nothing, Integer} = nothing,
                          bucket_s::Real = 3600)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    bucket_ms = max(1, Int(round(bucket_s * 1000)))
    return _rows(db, """
        SELECT (ts_epoch_ms / ?) * ? AS bucket_start_ms, COUNT(*) AS n
        FROM triggers
        WHERE ts_epoch_ms >= ? AND ts_epoch_ms <= ?
        GROUP BY bucket_start_ms
        ORDER BY bucket_start_ms ASC
        """,
        (bucket_ms, bucket_ms, Int(since), until_ms))
end

# ---------------------------------------------------------------------------
# Embedding scatter (2-D UMAP projection of persisted per-line vectors).
# ---------------------------------------------------------------------------

"""
    embedding_scatter(db; since, until = nothing, limit = 5000,
                      n_neighbors = 15, min_dist = 0.1,
                      random_state = nothing) -> Vector{NT}

Project the persisted per-line embeddings (`stream --persist-embeddings`)
inside `[since, until]` down to 2-D with UMAP. Returns one row per line:
`(line_id, x, y, cluster)`, where `cluster` is the line's
`drain_cluster_id` (or `0` when none was recorded) — ready to plot.

Returns `[]` when the window holds too few embedded lines to run UMAP
(it needs at least `n_neighbors` samples). Loads PythonCall lazily via
`Pipeline.umap_reduce`, so this stays an offline-analytics call — the
streaming hot path never touches it.
"""
function embedding_scatter(db::DB;
                           since::Integer,
                           until::Union{Nothing, Integer} = nothing,
                           limit::Integer = 5000,
                           n_neighbors::Integer = 15,
                           min_dist::Real = 0.1,
                           random_state = nothing)
    rows = lines_with_embeddings(db; since = since, until = until, limit = limit)
    length(rows) < max(2, n_neighbors) && return NamedTuple[]
    dims = length(rows[1].embedding)
    (dims > 0 && all(r -> length(r.embedding) == dims, rows)) ||
        throw(ArgumentError("persisted embeddings have inconsistent dimensions"))
    # (features × samples) as the rest of the pipeline expects.
    X = Matrix{Float64}(undef, dims, length(rows))
    @inbounds for (j, r) in enumerate(rows)
        X[:, j] = r.embedding
    end
    Y = umap_reduce(X; n_neighbors = Int(n_neighbors),
                    min_dist = Float64(min_dist), n_components = 2,
                    random_state = random_state)      # (2 × samples)
    return [(line_id = rows[j].line_id,
             x = Y[1, j], y = Y[2, j],
             cluster = rows[j].drain_cluster_id === nothing ? 0 :
                       Int(rows[j].drain_cluster_id))
            for j in eachindex(rows)]
end

end # module Insights
