"""
    Memory.SQLite

Long-term durable store for the streaming CLI. Holds triggers,
optional sampled per-line records, sessions, user-curated patterns,
pattern-match history, and free-form annotations.

The hot write path on the `stream` daemon goes through
[`spawn_writer`], which returns a bounded `Channel{WriteOp}` and a
background task that batches inserts inside a single transaction
every `flush_ms` ms (or `batch` ops, whichever trips first). The
inference loop never blocks on disk I/O.

Reader subcommands (`query`, `insights`, `report`, the TUI) call the
query helpers directly — they're plain prepared statements over WAL,
so a concurrent writer doesn't block them.

## Surface

- [`open_db`] / [`migrate!`] / [`schema_version`] — connection setup.
- [`spawn_writer`] — async batched writer for the stream loop.
- `insert_*` — synchronous one-shot writers (reader subcommands, tests).
- `triggers` / `top_rules` / `novel_clusters` / `cluster_timeline` /
  `cluster_id_sequence` — query helpers consumed by `insights`.
"""
module SQLiteStore

using SQLite: SQLite, DB
using ..Schema: Schema
using ...StructuredLog: StructuredLog
using JSON3
using Dates: Dates, DateTime, UTC, now, datetime2unix

# SQLite.jl cursors are lazy + forward-only. Use these helpers
# everywhere we issue a write or read so the side effects realise and
# the row values aren't lost between iterations.
@inline _exec!(db::DB, sql::AbstractString, params) =
    (foreach(identity, SQLite.DBInterface.execute(db, sql, params)); nothing)

# Materialise a Query's rows into a Vector{NamedTuple} so callers can
# pass results around without worrying about cursor lifetime.
function _rows(db::DB, sql::AbstractString, params = ())
    cursor = SQLite.DBInterface.execute(db, sql, params)
    out = NamedTuple[]
    for r in cursor
        push!(out, NamedTuple(pairs(r)))
    end
    return out
end

export open_db, migrate!, schema_version, spawn_writer, try_put!,
       WriterStats,
       insert_session!, finalize_session!, insert_trigger!,
       insert_line!, insert_pattern_match!,
       triggers, top_rules, novel_clusters, cluster_timeline,
       cluster_id_sequence, lines_with_embeddings,
       embedding_to_blob, blob_to_embedding,
       WriteOp, WriteSession, WriteTrigger, WriteLine, WritePatternMatch

# ---------------------------------------------------------------------------
# Embedding BLOB codec — a per-line vector is stored as raw
# little-endian Float32 bytes (schema v3 `lines.embedding`).
# ---------------------------------------------------------------------------

"Serialise an embedding vector to a raw Float32 byte blob."
embedding_to_blob(v::AbstractVector{<:Real}) =
    collect(reinterpret(UInt8, Vector{Float32}(v)))

"Decode a Float32 byte blob back to a `Vector{Float32}` (empty on `nothing`)."
blob_to_embedding(b::AbstractVector{UInt8}) =
    collect(reinterpret(Float32, Vector{UInt8}(b)))
blob_to_embedding(::Nothing) = Float32[]

# ---------------------------------------------------------------------------
# Connection setup.
# ---------------------------------------------------------------------------

"""
    open_db(path; create = true, wal = true) -> DB

Open a SQLite connection. `wal = true` switches to WAL journaling so
the reader subcommands don't block the writer. The file is created
when `create = true` and missing.
"""
function open_db(path::AbstractString; create::Bool = true, wal::Bool = true)
    !create && !isfile(path) && error("SQLite db not found: $path")
    db = SQLite.DB(String(path))
    if wal
        SQLite.execute(db, "PRAGMA journal_mode = WAL")
        SQLite.execute(db, "PRAGMA synchronous = NORMAL")
    end
    SQLite.execute(db, "PRAGMA foreign_keys = ON")
    return db
end

"""
    migrate!(db::DB) -> Int

Apply pending migrations. Returns the resulting head version.
"""
migrate!(db::DB) = Schema.apply_migrations!(db)

"""
    schema_version(db::DB) -> Int

Current schema head. `0` for a fresh / unmigrated DB.
"""
schema_version(db::DB) = Schema.current_version(db)

# ---------------------------------------------------------------------------
# Time helpers.
# ---------------------------------------------------------------------------

_now_utc() = now(UTC)

_iso(dt::DateTime) = string(Dates.format(dt,
                            Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

_epoch_ms(dt::DateTime) = Int(round(datetime2unix(dt) * 1000))

# Parse an ISO-8601 / naive DateTime string back to a DateTime; falls
# back to `now()` so a malformed input doesn't crash the writer.
function _parse_dt(s::AbstractString)
    try
        return DateTime(strip(replace(String(s), "Z" => "")),
                        Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss")
    catch
        try
            return DateTime(strip(replace(String(s), "Z" => "")))
        catch
            return _now_utc()
        end
    end
end

# ---------------------------------------------------------------------------
# Synchronous writers.
# ---------------------------------------------------------------------------

"""
    insert_session!(db; host, model_path, detector_path, rules_path,
                    rules_sha256, meta) -> Int

Insert a row in `sessions`, return the new id. `ended_at` /
`exit_code` are filled by [`finalize_session!`].
"""
function insert_session!(db::DB;
                         host::AbstractString = "",
                         model_path::AbstractString = "",
                         detector_path::AbstractString = "",
                         rules_path::AbstractString = "",
                         rules_sha256::AbstractString = "",
                         meta::AbstractDict = Dict{String, Any}())
    started = _iso(_now_utc())
    _exec!(db,
        "INSERT INTO sessions (started_at, host, model_path, detector_path, " *
        "rules_path, rules_sha256, meta_json) " *
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (started, host, model_path, detector_path, rules_path,
         rules_sha256, JSON3.write(meta)))
    return Int(SQLite.last_insert_rowid(db))
end

"""
    finalize_session!(db, session_id; exit_code = 0)

Patch a session row with the end timestamp + exit code.
"""
function finalize_session!(db::DB, session_id::Integer; exit_code::Integer = 0)
    _exec!(db,
        "UPDATE sessions SET ended_at = ?, exit_code = ? WHERE id = ?",
        (_iso(_now_utc()), Int(exit_code), Int(session_id)))
    return nothing
end

"""
    insert_trigger!(db; session_id, trigger, ir) -> Int

Persist one `TriggerEvent` along with the relevant slices of the
`InferResult` dict (frame, drain, model signals). Returns the new
row id.
"""
function insert_trigger!(db::DB;
                         session_id::Integer,
                         rule_id::AbstractString,
                         rule_kind::AbstractString,
                         severity::AbstractString,
                         line_id::Integer,
                         line::AbstractString,
                         fields::AbstractDict,
                         ts::Union{DateTime, AbstractString} = _now_utc(),
                         frame = nothing,
                         drain_cluster_id::Union{Nothing, Integer} = nothing,
                         model_signals = nothing)
    dt = ts isa DateTime ? ts : _parse_dt(ts)
    _exec!(db,
        "INSERT INTO triggers (session_id, ts, ts_epoch_ms, rule_id, rule_kind, " *
        "severity, line_id, line, fields_json, frame_json, drain_cluster_id, " *
        "model_signals_json) " *
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (Int(session_id), _iso(dt), _epoch_ms(dt),
         String(rule_id), String(rule_kind), String(severity),
         Int(line_id), String(line),
         JSON3.write(fields),
         frame === nothing ? nothing : JSON3.write(frame),
         drain_cluster_id === nothing ? nothing : Int(drain_cluster_id),
         model_signals === nothing ? nothing : JSON3.write(model_signals)))
    return Int(SQLite.last_insert_rowid(db))
end

"""
    insert_line!(db; session_id, line_id, line, ts, drain_cluster_id,
                 model_signals) -> Int
"""
function insert_line!(db::DB;
                      session_id::Integer,
                      line_id::Integer,
                      line::AbstractString,
                      ts::Union{DateTime, AbstractString} = _now_utc(),
                      drain_cluster_id::Union{Nothing, Integer} = nothing,
                      model_signals = nothing,
                      embedding::Union{Nothing, AbstractVector{<:Real}} = nothing)
    dt = ts isa DateTime ? ts : _parse_dt(ts)
    dcid = drain_cluster_id === nothing ? nothing : Int(drain_cluster_id)
    sig  = model_signals === nothing ? nothing : JSON3.write(model_signals)
    if embedding === nothing
        # Omit the embedding column entirely so this insert works against
        # any schema ≥ v1 (the column only exists from v3). The store
        # always migrates to HEAD in production, but tests exercise the
        # pre-migration schemas directly.
        _exec!(db,
            "INSERT INTO lines (line_id, session_id, ts, ts_epoch_ms, line, " *
            "drain_cluster_id, model_signals_json) " *
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (Int(line_id), Int(session_id), _iso(dt), _epoch_ms(dt),
             String(line), dcid, sig))
    else
        _exec!(db,
            "INSERT INTO lines (line_id, session_id, ts, ts_epoch_ms, line, " *
            "drain_cluster_id, model_signals_json, embedding) " *
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (Int(line_id), Int(session_id), _iso(dt), _epoch_ms(dt),
             String(line), dcid, sig, embedding_to_blob(embedding)))
    end
    return Int(line_id)
end

"""
    lines_with_embeddings(db; since, until = nothing, limit = 5000) -> Vector{NT}

Persisted line rows in `[since, until]` that carry an embedding blob.
Each row: `(line_id, ts_epoch_ms, drain_cluster_id, embedding::Vector{Float32})`.
"""
function lines_with_embeddings(db::DB;
                               since::Integer,
                               until::Union{Nothing, Integer} = nothing,
                               limit::Integer = 5000)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    raw = _rows(db,
        "SELECT line_id, ts_epoch_ms, drain_cluster_id, embedding " *
        "FROM lines " *
        "WHERE embedding IS NOT NULL AND ts_epoch_ms >= ? AND ts_epoch_ms <= ? " *
        "ORDER BY ts_epoch_ms ASC LIMIT ?",
        (Int(since), until_ms, Int(limit)))
    return [(line_id = Int(r.line_id),
             ts_epoch_ms = Int(r.ts_epoch_ms),
             drain_cluster_id = r.drain_cluster_id === missing ? nothing :
                                r.drain_cluster_id,
             embedding = blob_to_embedding(r.embedding)) for r in raw]
end

"""
    insert_pattern_match!(db; pattern_id, trigger_id, session_id,
                          line_id, ts, evidence) -> Int
"""
function insert_pattern_match!(db::DB;
                               pattern_id::Integer,
                               trigger_id::Union{Nothing, Integer} = nothing,
                               session_id::Integer,
                               line_id::Integer,
                               ts::Union{DateTime, AbstractString} = _now_utc(),
                               evidence = nothing)
    dt = ts isa DateTime ? ts : _parse_dt(ts)
    _exec!(db,
        "INSERT INTO pattern_matches (pattern_id, trigger_id, session_id, " *
        "ts_epoch_ms, line_id, evidence_json) " *
        "VALUES (?, ?, ?, ?, ?, ?)",
        (Int(pattern_id),
         trigger_id === nothing ? nothing : Int(trigger_id),
         Int(session_id), _epoch_ms(dt), Int(line_id),
         evidence === nothing ? nothing : JSON3.write(evidence)))
    return Int(SQLite.last_insert_rowid(db))
end

# ---------------------------------------------------------------------------
# Async batched writer.
# ---------------------------------------------------------------------------

abstract type WriteOp end

struct WriteSession        <: WriteOp; payload::Dict{Symbol, Any} end
struct WriteTrigger        <: WriteOp; payload::Dict{Symbol, Any} end
struct WriteLine           <: WriteOp; payload::Dict{Symbol, Any} end
struct WritePatternMatch   <: WriteOp; payload::Dict{Symbol, Any} end

"""
    WriterStats

Mutable counters mirrored into the streaming status heartbeat.
`dropped_ops` counts individual write ops that failed inside a
batch (e.g. a constraint violation) — the failing op is discarded,
the rest of the batch still commits, and the writer stays alive.
"""
Base.@kwdef mutable struct WriterStats
    flushed_total::Int = 0
    dropped_ops::Int   = 0
    flush_errors::Int  = 0
end

"""
    spawn_writer(db; batch = 100, flush_ms = 250, capacity = 4096)
        -> (channel, task, stop_ref, stats)

Spawn an async task that drains `channel` and bulk-inserts the
batch inside a single `BEGIN IMMEDIATE … COMMIT` block. The producer
calls `put!(channel, WriteTrigger(...))` etc.; the writer flushes
when either `batch` ops are queued or `flush_ms` ms have elapsed.

Failure containment: each op is applied inside its own try/catch,
so one bad row (constraint violation, malformed payload) is dropped
and counted on `stats.dropped_ops` while the rest of the batch
commits. No exception can kill the writer task.

Clean shutdown: set `stop_ref[] = true` and `close(channel)`; the
writer drains, commits, and exits. Closing the channel alone also
terminates the loop once drained (no busy spin).
"""
function spawn_writer(db::DB; batch::Int = 100, flush_ms::Int = 250,
                      capacity::Int = 4096)
    ch = Channel{WriteOp}(capacity)
    stop = Ref(false)
    stats = WriterStats()
    task = @async _writer_loop(db, ch, batch, flush_ms, stop, stats)
    return (ch, task, stop, stats)
end

"""
    try_put!(ch::Channel, op) -> Bool

Non-blocking put for a **single-producer** channel: returns `false`
instead of blocking when the channel is full, and `false` (instead
of throwing) when it is closed. The capacity check is check-then-act,
which is only race-free because exactly one task produces onto the
writer channel — do not share the producing side across tasks.
"""
function try_put!(ch::Channel, op)
    isopen(ch) || return false
    Base.n_avail(ch) >= ch.sz_max && return false
    try
        put!(ch, op)
        return true
    catch
        return false
    end
end

function _writer_loop(db::DB, ch::Channel{WriteOp}, batch::Int,
                      flush_ms::Int, stop::Ref{Bool}, stats::WriterStats)
    pending = WriteOp[]
    last_flush = time()
    try
        while true
            op = _try_take!(ch, max(0.001, flush_ms / 1000))
            if op !== nothing
                push!(pending, op)
            end
            should_flush = !isempty(pending) && (
                length(pending) >= batch ||
                (time() - last_flush) * 1000 >= flush_ms ||
                op === nothing
            )
            if should_flush
                _flush_batch!(db, pending, stats)
                empty!(pending)
                last_flush = time()
            end
            # Exit when: told to stop and drained, or channel closed
            # and drained. The closed-and-drained arm prevents the
            # zero-sleep spin a bare `close(ch)` used to cause.
            drained = !isready(ch) && isempty(pending)
            (stop[] || !isopen(ch)) && drained && break
        end
    catch e
        # Should be unreachable — _flush_batch! contains its errors —
        # but never let the writer die silently if it happens.
        StructuredLog.error_event("memory writer crashed";
                                   error = sprint(showerror, e))
    finally
        if !isempty(pending)
            try
                _flush_batch!(db, pending, stats)
            catch
            end
        end
    end
end

function _try_take!(ch::Channel, timeout::Float64)
    isready(ch) && return take!(ch)
    deadline = time() + timeout
    while !isready(ch) && time() < deadline && isopen(ch)
        sleep(0.005)
    end
    isready(ch) && return take!(ch)
    return nothing
end

function _flush_batch!(db::DB, ops::Vector{WriteOp}, stats::WriterStats)
    isempty(ops) && return
    try
        SQLite.transaction(db) do
            for op in ops
                try
                    _apply_op!(db, op)
                catch e
                    # Drop the failing op, keep the batch. Constraint
                    # violations land here; the transaction survives
                    # because the statement-level error doesn't abort
                    # SQLite's enclosing transaction.
                    stats.dropped_ops += 1
                    StructuredLog.error_event("memory write dropped";
                        op = string(typeof(op)),
                        error = sprint(showerror, e))
                end
            end
        end
        stats.flushed_total += length(ops)
    catch e
        # Transaction-level failure (disk full, db locked beyond
        # busy timeout). Count and drop the batch — durability is
        # best-effort; stdout remains the primary sink.
        stats.flush_errors += 1
        stats.dropped_ops += length(ops)
        StructuredLog.error_event("memory batch flush failed";
            n_ops = length(ops),
            error = sprint(showerror, e))
    end
end

function _apply_op!(db::DB, op::WriteTrigger)
    insert_trigger!(db; op.payload...)
end
function _apply_op!(db::DB, op::WriteLine)
    insert_line!(db; op.payload...)
end
function _apply_op!(db::DB, op::WritePatternMatch)
    insert_pattern_match!(db; op.payload...)
end
function _apply_op!(db::DB, op::WriteSession)
    # Sessions are inserted synchronously at boot; this op is for
    # finalising on shutdown.
    finalize_session!(db, Int(op.payload[:session_id]);
                      exit_code = Int(get(op.payload, :exit_code, 0)))
end

# ---------------------------------------------------------------------------
# Query helpers (used by `query`, `insights`, `report`, the TUI).
# ---------------------------------------------------------------------------

"""
    triggers(db; since, until, rule_id, severity, cluster_id, limit) -> Vector{NamedTuple}

Read triggers in a time window with optional filters. All four
filter args are optional; `nothing` (the default) means no filter on
that column.
"""
function triggers(db::DB;
                  since::Union{Nothing, Integer} = nothing,
                  until::Union{Nothing, Integer} = nothing,
                  rule_id::Union{Nothing, AbstractString} = nothing,
                  severity::Union{Nothing, AbstractString} = nothing,
                  cluster_id::Union{Nothing, Integer} = nothing,
                  limit::Integer = 100)
    where_clauses = String[]
    params = Any[]
    if since !== nothing
        push!(where_clauses, "ts_epoch_ms >= ?")
        push!(params, Int(since))
    end
    if until !== nothing
        push!(where_clauses, "ts_epoch_ms <= ?")
        push!(params, Int(until))
    end
    if rule_id !== nothing
        push!(where_clauses, "rule_id = ?")
        push!(params, String(rule_id))
    end
    if severity !== nothing
        push!(where_clauses, "severity = ?")
        push!(params, String(severity))
    end
    if cluster_id !== nothing
        push!(where_clauses, "drain_cluster_id = ?")
        push!(params, Int(cluster_id))
    end
    where_sql = isempty(where_clauses) ? "" :
        " WHERE " * join(where_clauses, " AND ")
    sql = "SELECT id, session_id, ts, ts_epoch_ms, rule_id, rule_kind, " *
          "severity, line_id, line, fields_json, frame_json, " *
          "drain_cluster_id, model_signals_json " *
          "FROM triggers" * where_sql *
          " ORDER BY ts_epoch_ms DESC LIMIT ?"
    push!(params, Int(limit))
    return _rows(db, sql, params)
end

"""
    top_rules(db; since, until, limit) -> Vector{NamedTuple}

Trigger counts grouped by `rule_id`, descending.
"""
function top_rules(db::DB;
                   since::Union{Nothing, Integer} = nothing,
                   until::Union{Nothing, Integer} = nothing,
                   limit::Integer = 10)
    where_clauses = String[]
    params = Any[]
    if since !== nothing
        push!(where_clauses, "ts_epoch_ms >= ?")
        push!(params, Int(since))
    end
    if until !== nothing
        push!(where_clauses, "ts_epoch_ms <= ?")
        push!(params, Int(until))
    end
    where_sql = isempty(where_clauses) ? "" :
        " WHERE " * join(where_clauses, " AND ")
    sql = "SELECT rule_id, severity, COUNT(*) AS n, " *
          "MAX(ts_epoch_ms) AS last_seen " *
          "FROM triggers" * where_sql *
          " GROUP BY rule_id, severity ORDER BY n DESC LIMIT ?"
    push!(params, Int(limit))
    return _rows(db, sql, params)
end

"""
    novel_clusters(db; since, until = nothing) -> Vector{NamedTuple}

Drain cluster ids that first appeared in `[since, until]`. A cluster
counts as "novel in the window" when its first sighting in `lines`
or `triggers` falls inside the window.
"""
function novel_clusters(db::DB;
                        since::Integer,
                        until::Union{Nothing, Integer} = nothing)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    sql = """
    SELECT cluster_id, first_seen FROM (
        SELECT drain_cluster_id AS cluster_id,
               MIN(ts_epoch_ms)  AS first_seen
        FROM triggers
        WHERE drain_cluster_id IS NOT NULL
        GROUP BY drain_cluster_id
    )
    WHERE first_seen >= ? AND first_seen <= ?
    ORDER BY first_seen ASC
    """
    return _rows(db, sql, (Int(since), until_ms))
end

"""
    cluster_timeline(db; cluster_id, since, until = nothing, bucket_s = 60)
        -> Vector{NamedTuple}

Bucketed counts of trigger events involving `cluster_id`. Each row
is `(bucket_start_ms, n)`.
"""
function cluster_timeline(db::DB;
                          cluster_id::Integer,
                          since::Integer,
                          until::Union{Nothing, Integer} = nothing,
                          bucket_s::Real = 60)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    bucket_ms = Int(round(bucket_s * 1000))
    sql = """
    SELECT (ts_epoch_ms / ?) * ? AS bucket_start_ms, COUNT(*) AS n
    FROM triggers
    WHERE drain_cluster_id = ? AND ts_epoch_ms >= ? AND ts_epoch_ms <= ?
    GROUP BY bucket_start_ms
    ORDER BY bucket_start_ms ASC
    """
    return _rows(db, sql,
        (bucket_ms, bucket_ms, Int(cluster_id), Int(since), until_ms))
end

"""
    cluster_id_sequence(db; since, until = nothing) -> Vector{Int}

Time-ordered sequence of `drain_cluster_id` values for rows in
`lines` (preferred) or `triggers`, used as input to
`Mining.Episodes.mv_span` from the `insights --episodes` subcommand.
"""
function cluster_id_sequence(db::DB;
                              since::Integer,
                              until::Union{Nothing, Integer} = nothing)
    until_ms = until === nothing ? typemax(Int) : Int(until)
    # Pull from `lines` when populated; otherwise fall back to `triggers`.
    cnt_rows = _rows(db,
        "SELECT COUNT(*) AS n FROM lines WHERE drain_cluster_id IS NOT NULL")
    n_lines = isempty(cnt_rows) ? 0 : something(first(cnt_rows).n, 0)
    table = n_lines > 0 ? "lines" : "triggers"
    sql = "SELECT drain_cluster_id FROM $table " *
          "WHERE drain_cluster_id IS NOT NULL " *
          "AND ts_epoch_ms >= ? AND ts_epoch_ms <= ? " *
          "ORDER BY ts_epoch_ms ASC"
    rows = _rows(db, sql, (Int(since), until_ms))
    return Int[Int(r.drain_cluster_id) for r in rows]
end

# ---------------------------------------------------------------------------
# Convenience.
# ---------------------------------------------------------------------------

"""
    epoch_ms_since(s::AbstractString) -> Int

Parse a relative duration (`24h`, `7d`, `30m`, `45s`) or an ISO-8601
timestamp into an epoch-ms cutoff.
"""
function epoch_ms_since(s::AbstractString)
    s = strip(s)
    if !isempty(s) && s[end] in ('s', 'm', 'h', 'd')
        unit = s[end]
        n = tryparse(Float64, s[1:end-1])
        n === nothing && throw(ArgumentError(
            "invalid --since `$s`: expected e.g. 24h, 7d, 30m, 45s, or an ISO-8601 timestamp"))
        mul = unit == 's' ? 1_000 :
              unit == 'm' ? 60_000 :
              unit == 'h' ? 3_600_000 :
                            86_400_000
        return _epoch_ms(_now_utc()) - Int(round(n * mul))
    end
    # Absolute timestamp — must parse cleanly. A silent "now" fallback
    # would turn a typo'd offset (e.g. "...+02:00") into an empty result.
    dt = _try_parse_dt(s)
    dt === nothing && throw(ArgumentError(
        "invalid --since `$s`: expected e.g. 24h, 7d, 30m, 45s, or an ISO-8601 " *
        "timestamp without a UTC offset (times are treated as UTC)"))
    return _epoch_ms(dt)
end

"""
    iso_from_epoch_ms(ms) -> String

Render an epoch-ms cutoff back to the `_iso` TEXT form so it can be
compared lexicographically against the ISO `started_at` column
(ISO-8601 sorts in chronological order for a fixed format).
"""
iso_from_epoch_ms(ms::Integer) =
    _iso(Dates.unix2datetime(ms / 1000))

# Strict variant of _parse_dt used by epoch_ms_since: returns nothing
# instead of falling back to now() so callers can raise a clear error.
function _try_parse_dt(s::AbstractString)
    for fmt in (Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss",
                Dates.dateformat"yyyy-mm-ddTHH:MM:SS")
        try
            return DateTime(strip(replace(String(s), "Z" => "")), fmt)
        catch
        end
    end
    try
        return DateTime(strip(replace(String(s), "Z" => "")))
    catch
        return nothing
    end
end

end # module SQLiteStore
