"""
    Memory.Schema

Versioned SQLite migrations for the long-term store. Each migration
is a `(version, sql)` pair applied in order on first open. The
`schema_version` table holds the current head; out-of-band tampering
is detected at boot.

The head version is [`HEAD`]; bump it when adding a migration.
"""
module Schema

export HEAD, MIGRATIONS, apply_migrations!, current_version

using SQLite: SQLite, DB

# DBInterface.execute returns a lazy cursor for SQLite.jl — INSERTs /
# UPDATEs only fire when the cursor is iterated. `_exec!` is a thin
# wrapper that drains the cursor so the side effect is realised
# immediately. Use this for every parameterised write.
@inline _exec!(db::DB, sql::AbstractString, params) =
    (foreach(identity, SQLite.DBInterface.execute(db, sql, params)); nothing)

const HEAD = 1

"""
    MIGRATIONS

Ordered list of `(version, sql)` tuples. Each `sql` block may contain
multiple statements separated by `;` — they're executed one at a time
so SQLite's prepare-each-statement requirement is satisfied.
"""
const MIGRATIONS = [
    (1, raw"""
    CREATE TABLE schema_version (version INTEGER NOT NULL);

    CREATE TABLE sessions (
        id            INTEGER PRIMARY KEY,
        started_at    TEXT    NOT NULL,
        ended_at      TEXT,
        host          TEXT,
        model_path    TEXT,
        detector_path TEXT,
        rules_path    TEXT,
        rules_sha256  TEXT,
        exit_code     INTEGER,
        meta_json     TEXT
    );

    CREATE TABLE triggers (
        id                 INTEGER PRIMARY KEY,
        session_id         INTEGER REFERENCES sessions(id),
        ts                 TEXT    NOT NULL,
        ts_epoch_ms        INTEGER NOT NULL,
        rule_id            TEXT    NOT NULL,
        rule_kind          TEXT    NOT NULL,
        severity           TEXT    NOT NULL,
        line_id            INTEGER NOT NULL,
        line               TEXT    NOT NULL,
        fields_json        TEXT    NOT NULL,
        frame_json         TEXT,
        drain_cluster_id   INTEGER,
        model_signals_json TEXT
    );

    CREATE INDEX idx_triggers_ts      ON triggers(ts_epoch_ms);
    CREATE INDEX idx_triggers_rule_ts ON triggers(rule_id, ts_epoch_ms);
    CREATE INDEX idx_triggers_sev_ts  ON triggers(severity, ts_epoch_ms);
    CREATE INDEX idx_triggers_cluster ON triggers(drain_cluster_id, ts_epoch_ms);

    CREATE TABLE lines (
        line_id            INTEGER PRIMARY KEY,
        session_id         INTEGER REFERENCES sessions(id),
        ts                 TEXT    NOT NULL,
        ts_epoch_ms        INTEGER NOT NULL,
        line               TEXT    NOT NULL,
        drain_cluster_id   INTEGER,
        model_signals_json TEXT
    );

    CREATE INDEX idx_lines_ts      ON lines(ts_epoch_ms);
    CREATE INDEX idx_lines_cluster ON lines(drain_cluster_id, ts_epoch_ms);

    CREATE TABLE patterns (
        id                       INTEGER PRIMARY KEY,
        name                     TEXT    NOT NULL UNIQUE,
        description              TEXT,
        severity                 TEXT    NOT NULL DEFAULT 'warn',
        cooldown_s               REAL    NOT NULL DEFAULT 60,
        match_kind               TEXT    NOT NULL,
        match_drain_template_id  INTEGER,
        match_regex              TEXT,
        match_keywords_json      TEXT,
        case_sensitive           INTEGER NOT NULL DEFAULT 0,
        created_by               TEXT,
        created_at               TEXT    NOT NULL,
        enabled                  INTEGER NOT NULL DEFAULT 1
    );

    CREATE INDEX idx_patterns_enabled ON patterns(enabled);

    CREATE TABLE pattern_matches (
        id            INTEGER PRIMARY KEY,
        pattern_id    INTEGER NOT NULL REFERENCES patterns(id),
        trigger_id    INTEGER REFERENCES triggers(id),
        session_id    INTEGER REFERENCES sessions(id),
        ts_epoch_ms   INTEGER NOT NULL,
        line_id       INTEGER NOT NULL,
        evidence_json TEXT
    );

    CREATE INDEX idx_pmatch_ts      ON pattern_matches(ts_epoch_ms);
    CREATE INDEX idx_pmatch_pattern ON pattern_matches(pattern_id, ts_epoch_ms);

    CREATE TABLE annotations (
        id          INTEGER PRIMARY KEY,
        target_kind TEXT    NOT NULL,
        target_id   INTEGER NOT NULL,
        note        TEXT    NOT NULL,
        author      TEXT,
        ts          TEXT    NOT NULL
    );

    CREATE INDEX idx_annot_target ON annotations(target_kind, target_id);
    """),
]

"""
    current_version(db::DB) -> Int

Read the head version from `schema_version`. Returns 0 when the
table doesn't exist yet (fresh DB).
"""
function current_version(db::DB)
    # SQLite.jl Query rows are forward-only — collecting them only
    # captures references unless we materialise each row into a
    # NamedTuple as we iterate.
    rows = SQLite.DBInterface.execute(db,
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
    has_table = false
    for _ in rows
        has_table = true
        break
    end
    has_table || return 0
    rows = SQLite.DBInterface.execute(db, "SELECT version FROM schema_version")
    v = nothing
    for r in rows
        v = r.version
        break
    end
    v === nothing && return 0
    v isa Missing && return 0
    return Int(v)
end

"""
    apply_migrations!(db::DB)

Apply every migration above the current version, in order, inside a
single transaction. Bumps `schema_version` after each. Idempotent.
"""
function apply_migrations!(db::DB)
    cur = current_version(db)
    # Use explicit BEGIN/COMMIT so DDL + the version bookkeeping land
    # atomically; SQLite.transaction has flaky semantics around DDL.
    SQLite.execute(db, "BEGIN")
    try
        for (v, sql) in MIGRATIONS
            v <= cur && continue
            for stmt in _split_statements(sql)
                isempty(strip(stmt)) && continue
                SQLite.execute(db, stmt)
            end
            if cur == 0 && v == 1
                _exec!(db, "INSERT INTO schema_version (version) VALUES (?)", (v,))
            else
                _exec!(db, "UPDATE schema_version SET version = ?", (v,))
            end
            cur = v
        end
        SQLite.execute(db, "COMMIT")
    catch
        try; SQLite.execute(db, "ROLLBACK"); catch; end
        rethrow()
    end
    return current_version(db)
end

# SQLite.execute takes one statement at a time; chop on ';' but be
# careful with strings that contain semicolons. The migrations above
# are plain DDL with no embedded semicolons, so a naive split is safe.
function _split_statements(sql::AbstractString)
    return split(sql, ';')
end

end # module Schema
