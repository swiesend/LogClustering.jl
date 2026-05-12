using Test
using LogClustering
using LogClustering.Memory.SQLite
using LogClustering.Memory.SQLite: open_db, migrate!, schema_version,
                                    spawn_writer, insert_session!,
                                    finalize_session!, insert_trigger!,
                                    insert_line!, insert_pattern_match!,
                                    triggers, top_rules, novel_clusters,
                                    cluster_timeline, cluster_id_sequence,
                                    WriteTrigger, WriteLine,
                                    WritePatternMatch
using LogClustering.Memory.SQLite: epoch_ms_since
using LogClustering.Memory.Schema: HEAD
using SQLite: SQLite
using Dates: Dates, DateTime, now, UTC

@testset "Memory.SQLite" begin

    @testset "open_db + migrate! brings a fresh file to HEAD" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            @test schema_version(db) == 0
            v = migrate!(db)
            @test v == HEAD
            @test schema_version(db) == HEAD
            # Idempotent.
            @test migrate!(db) == HEAD
        end
    end

    @testset "session + trigger round-trip" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)

            sid = insert_session!(db; host = "srv-1",
                                  model_path = "m.jld2",
                                  rules_path = "r.json")
            @test sid isa Int && sid > 0

            tid = insert_trigger!(db; session_id = sid,
                                  rule_id = "fatal", rule_kind = "keyword",
                                  severity = "crit", line_id = 7,
                                  line = "FATAL: oops",
                                  fields = Dict("matched" => ["FATAL"]),
                                  drain_cluster_id = 12)
            @test tid > 0

            finalize_session!(db, sid; exit_code = 0)

            rows = triggers(db; limit = 10)
            @test length(rows) == 1
            r = rows[1]
            @test String(r.rule_id) == "fatal"
            @test Int(r.line_id)    == 7
            @test Int(r.drain_cluster_id) == 12
            @test occursin("FATAL", String(r.fields_json))
        end
    end

    @testset "async writer batches inserts into one transaction" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")

            ch, task, stop = spawn_writer(db; batch = 50, flush_ms = 30)
            try
                for i in 1:200
                    put!(ch, WriteTrigger(Dict{Symbol, Any}(
                        :session_id => sid,
                        :rule_id    => "r$(i % 5)",
                        :rule_kind  => "keyword",
                        :severity   => i % 7 == 0 ? "crit" : "warn",
                        :line_id    => i,
                        :line       => "line $i",
                        :fields     => Dict("i" => i),
                        :drain_cluster_id => (i % 3) + 1,
                    )))
                end
            finally
                stop[] = true
                close(ch)
                # Bounded wait — the writer should drain in well under
                # a couple of seconds on a tmpfs path.
                timedwait(() -> istaskdone(task), 5.0)
            end

            rows = triggers(db; limit = 1000)
            @test length(rows) == 200
            top = top_rules(db; limit = 10)
            # 5 unique rule_ids x 2 severities (some rules have both warn + crit)
            # so we just assert we have at least 5 distinct rule_ids.
            @test length(unique(String(r.rule_id) for r in top)) == 5
        end
    end

    @testset "top_rules orders by count descending" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            for _ in 1:7
                insert_trigger!(db; session_id = sid,
                                rule_id = "a", rule_kind = "keyword",
                                severity = "warn", line_id = 1,
                                line = "x", fields = Dict())
            end
            for _ in 1:3
                insert_trigger!(db; session_id = sid,
                                rule_id = "b", rule_kind = "regex",
                                severity = "warn", line_id = 1,
                                line = "x", fields = Dict())
            end
            top = top_rules(db; limit = 5)
            @test String(top[1].rule_id) == "a"
            @test Int(top[1].n)          == 7
            @test String(top[2].rule_id) == "b"
            @test Int(top[2].n)          == 3
        end
    end

    @testset "novel_clusters reports first-seen ids within the window" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            # cluster 1 fired before the window; cluster 2 fires inside.
            insert_trigger!(db; session_id = sid, rule_id = "r",
                            rule_kind = "keyword", severity = "warn",
                            line_id = 1, line = "x",
                            fields = Dict(),
                            ts = DateTime("2024-01-01T00:00:00"),
                            drain_cluster_id = 1)
            insert_trigger!(db; session_id = sid, rule_id = "r",
                            rule_kind = "keyword", severity = "warn",
                            line_id = 2, line = "x",
                            fields = Dict(),
                            ts = DateTime("2026-04-27T14:00:00"),
                            drain_cluster_id = 2)
            since_ms = epoch_ms_since("2026-04-27T13:00:00")
            rows = novel_clusters(db; since = since_ms)
            @test length(rows) == 1
            @test Int(rows[1].cluster_id) == 2
        end
    end

    @testset "cluster_timeline buckets by minute" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            # Three events in the same minute, one in the next.
            base = DateTime("2026-04-27T14:00:00")
            for offset_s in (1, 5, 30, 65)
                insert_trigger!(db; session_id = sid, rule_id = "r",
                                rule_kind = "keyword", severity = "warn",
                                line_id = 1, line = "x",
                                fields = Dict(),
                                ts = base + Dates.Second(offset_s),
                                drain_cluster_id = 9)
            end
            rows = cluster_timeline(db;
                cluster_id = 9,
                since = epoch_ms_since("2026-04-27T13:00:00"),
                bucket_s = 60)
            @test length(rows) == 2
            @test sum(Int(r.n) for r in rows) == 4
        end
    end

    @testset "cluster_id_sequence pulls a time-ordered vector" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            base = DateTime("2026-04-27T14:00:00")
            for (i, cid) in enumerate([3, 1, 2, 1, 3])
                insert_trigger!(db; session_id = sid, rule_id = "r",
                                rule_kind = "keyword", severity = "warn",
                                line_id = i, line = "x",
                                fields = Dict(),
                                ts = base + Dates.Second(i),
                                drain_cluster_id = cid)
            end
            seq = cluster_id_sequence(db;
                since = epoch_ms_since("2026-04-27T13:00:00"))
            @test seq == [3, 1, 2, 1, 3]
        end
    end

    @testset "pattern_matches insert + fk integrity" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            # Seed a pattern row by hand (cmd_patterns isn't in this commit yet).
            foreach(identity, SQLite.DBInterface.execute(db,
                "INSERT INTO patterns (name, severity, match_kind, " *
                "match_keywords_json, created_at) VALUES (?, ?, ?, ?, ?)",
                ("p1", "warn", "keyword", "[\"x\"]", "2026-04-27")))
            pid = Int(SQLite.last_insert_rowid(db))
            mid = insert_pattern_match!(db; pattern_id = pid,
                                        session_id = sid,
                                        line_id = 42,
                                        evidence = Dict("matched" => ["x"]))
            @test mid > 0
        end
    end

    @testset "epoch_ms_since parses relative durations" begin
        ms_now = Int(round(time() * 1000))
        ms_24h = epoch_ms_since("24h")
        @test abs((ms_now - ms_24h) - 24 * 3_600_000) < 5_000
        ms_30m = epoch_ms_since("30m")
        @test abs((ms_now - ms_30m) - 30 * 60_000) < 5_000
        ms_iso = epoch_ms_since("2026-04-27T14:00:00.000")
        @test ms_iso > 0
    end

end
