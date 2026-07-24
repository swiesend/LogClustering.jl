using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Memory.SQLite
using LogClustering.Memory.SQLite: open_db, migrate!, insert_session!,
                                    insert_trigger!, insert_line!,
                                    epoch_ms_since
using LogClustering.Memory.Insights
using LogClustering.Memory.Insights: top_rules_window, novel_clusters_window,
                                      burstiness, cluster_view, transitions,
                                      episodes, pinned_summary, embedding_scatter
using LogClustering.Memory.PatternCatalog: pin_manual
using LogClustering.StructuredLog
using SQLite: SQLite
using JSON3
using Dates: Dates, DateTime

const _INS_LOG = IOBuffer()
StructuredLog.set_format!(:json; stream = _INS_LOG)

function _fill_fixture(path::AbstractString)
    db = open_db(path)
    migrate!(db)
    sid = insert_session!(db; host = "h")
    base = DateTime("2026-04-27T14:00:00")
    # Seed: rule "a" fires 5x in cluster 1; rule "b" 2x in cluster 2.
    # cluster 3 appears late so novel_clusters_window with a tight
    # `since` can pick it out.
    for i in 1:5
        insert_trigger!(db; session_id = sid,
            rule_id = "a", rule_kind = "keyword", severity = "warn",
            line_id = i, line = "L$i", fields = Dict("matched" => ["X"]),
            ts = base + Dates.Second(i),
            drain_cluster_id = 1)
    end
    for i in 6:7
        insert_trigger!(db; session_id = sid,
            rule_id = "b", rule_kind = "regex", severity = "crit",
            line_id = i, line = "L$i", fields = Dict(),
            ts = base + Dates.Second(i),
            drain_cluster_id = 2)
    end
    # Cluster 3 late-arrival.
    insert_trigger!(db; session_id = sid,
        rule_id = "a", rule_kind = "keyword", severity = "warn",
        line_id = 100, line = "Lnew", fields = Dict(),
        ts = base + Dates.Hour(2),
        drain_cluster_id = 3)
    # Populate lines with a clean cluster-id sequence for episode mining.
    for (i, cid) in enumerate([1, 2, 3, 1, 2, 3, 1, 2, 3])
        insert_line!(db; session_id = sid,
            line_id = 200 + i, line = "seq $i",
            ts = base + Dates.Second(60 + i),
            drain_cluster_id = cid)
    end
    return (db = db, sid = sid, base = base)
end

@testset "Memory.Insights" begin

    @testset "top_rules_window groups + orders by count" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            rows = top_rules_window(d.db;
                since = epoch_ms_since("2026-01-01T00:00:00"),
                limit = 10)
            @test length(rows) == 2
            @test String(rows[1].rule_id) == "a"
            @test Int(rows[1].n) == 6   # 5 in cluster 1 + 1 late
        end
    end

    @testset "novel_clusters_window catches the late-arrival cluster" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            # "since" = base + 1h → only the late cluster 3 should appear.
            since_ms = epoch_ms_since(string(d.base + Dates.Hour(1)))
            rows = novel_clusters_window(d.db; since = since_ms)
            @test length(rows) == 1
            @test Int(rows[1].cluster_id) == 3
        end
    end

    @testset "burstiness buckets by minute" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            rows = burstiness(d.db; rule_id = "a",
                since = epoch_ms_since("2026-01-01T00:00:00"),
                bucket_s = 60)
            # 5 in the first minute, 1 two hours later.
            @test length(rows) == 2
            @test sum(Int(r.n) for r in rows) == 6
        end
    end

    @testset "transitions counts bigrams over the cluster sequence" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            rows = transitions(d.db;
                since = epoch_ms_since("2026-01-01T00:00:00"),
                top = 10)
            # Sequence 1,2,3,1,2,3,1,2,3 → (1->2)x3, (2->3)x3, (3->1)x2.
            tbl = Dict((Int(r.from), Int(r.to)) => Int(r.n) for r in rows)
            @test tbl[(1, 2)] == 3
            @test tbl[(2, 3)] == 3
            @test tbl[(3, 1)] == 2
        end
    end

    @testset "episodes returns mined patterns sorted by support" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            rows = episodes(d.db;
                since = epoch_ms_since("2026-01-01T00:00:00"),
                min_sup = 2, max_gap = 5, max_time_duration = 50,
                top = 10)
            @test !isempty(rows)
            # Most supported patterns should appear first.
            for i in 1:length(rows) - 1
                @test Int(rows[i].support) >= Int(rows[i + 1].support)
            end
        end
    end

    @testset "pinned_summary joins with patterns table" begin
        mktempdir() do dir
            d = _fill_fixture(joinpath(dir, "t.sqlite"))
            pin_manual(d.db; name = "p1", match_kind = :keyword,
                       match_keywords = ["L1"])
            rows = pinned_summary(d.db;
                since = epoch_ms_since("2026-01-01T00:00:00"))
            @test length(rows) == 1
            @test String(rows[1].name) == "p1"
            @test Int(rows[1].n) == 0   # no pattern_matches inserted
        end
    end

    @testset "embedding_scatter — few rows short-circuit, UMAP lazy" begin
        mktempdir() do dir
            db = open_db(joinpath(dir, "t.sqlite"))
            migrate!(db)
            sid = insert_session!(db; host = "h")
            base = DateTime("2026-04-27T14:00:00")
            since = epoch_ms_since("2026-01-01T00:00:00")

            # Fewer than n_neighbors embedded rows → empty, no Python.
            for i in 1:5
                insert_line!(db; session_id = sid, line_id = i,
                    line = "L$i", ts = base + Dates.Second(i),
                    drain_cluster_id = (i % 3) + 1,
                    embedding = Float32[sin(i), cos(i), Float32(i)])
            end
            @test isempty(embedding_scatter(db; since = since, n_neighbors = 15))

            # Enough rows: UMAP loads PythonCall lazily. Without the uv
            # venv it must raise the clear py/uv bootstrap error (never a
            # cold-start import); with it present, it returns a row per
            # line tagged (line_id, x, y, cluster).
            for i in 6:40
                insert_line!(db; session_id = sid, line_id = i,
                    line = "L$i", ts = base + Dates.Second(i),
                    drain_cluster_id = (i % 3) + 1,
                    embedding = Float32[sin(i), cos(i), Float32(i % 5)])
            end
            err = try
                rows = embedding_scatter(db; since = since,
                                         n_neighbors = 10, random_state = 42)
                @test length(rows) == 40
                @test all(r -> r.cluster in (1, 2, 3), rows)
                @test all(r -> isfinite(r.x) && isfinite(r.y), rows)
                nothing
            catch e
                sprint(showerror, e)
            end
            if err !== nothing
                @test occursin("py/", err) || occursin("PythonCall", err) ||
                      occursin("uv", err)
            end
        end
    end

end

# ---------------------------------------------------------------------------
# CLI subcommands.
# ---------------------------------------------------------------------------

function _capture(f)
    orig_out, orig_err = stdout, stderr
    rr_out = Pipe(); Base.link_pipe!(rr_out, reader_supports_async = true,
                                     writer_supports_async = true)
    rr_err = Pipe(); Base.link_pipe!(rr_err, reader_supports_async = true,
                                     writer_supports_async = true)
    redirect_stdout(rr_out); redirect_stderr(rr_err)
    code = try; f(); finally
        redirect_stdout(orig_out); redirect_stderr(orig_err)
        close(rr_out.in); close(rr_err.in)
    end
    return (code = code, out = read(rr_out, String), err = read(rr_err, String))
end

@testset "CLI — query / insights / report" begin

    @testset "query triggers --json emits NDJSON" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            r = _capture(() -> CLI.main(["query", "triggers",
                "--memory", db_path, "--since", "1000d",
                "--limit", "100", "--json"]))
            @test r.code == 0
            lines = filter(!isempty, split(r.out, '\n'))
            @test length(lines) == 8
            # Each line is valid JSON.
            for l in lines
                d = JSON3.read(l)
                @test haskey(d, "rule_id")
            end
        end
    end

    @testset "query sessions --since actually filters (ISO comparison)" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            db = open_db(db_path); migrate!(db)
            # One old session (hand-inserted ISO), one recent.
            foreach(identity, SQLite.DBInterface.execute(db,
                "INSERT INTO sessions (started_at, host) VALUES (?, ?)",
                ("2020-01-01T00:00:00.000Z", "old")))
            insert_session!(db; host = "recent")
            # --since 1h must exclude the 2020 session.
            r = _capture(() -> CLI.main(["query", "sessions",
                "--memory", db_path, "--since", "1h", "--json"]))
            @test r.code == 0
            rows = [JSON3.read(l) for l in filter(!isempty, split(r.out, '\n'))]
            hosts = [String(x["host"]) for x in rows]
            @test "recent" in hosts
            @test !("old" in hosts)
        end
    end

    @testset "insights --top-rules --episodes --json composes views" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            r = _capture(() -> CLI.main(["insights",
                "--memory", db_path, "--since", "1000d",
                "--top-rules", "--episodes", "--min-sup", "2",
                "--max-gap", "5", "--json"]))
            @test r.code == 0
            d = JSON3.read(r.out)
            @test haskey(d, "top_rules")
            @test haskey(d, "episodes")
            @test length(d["top_rules"]) >= 1
        end
    end

    @testset "report --format md produces every section" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            r = _capture(() -> CLI.main(["report",
                "--memory", db_path, "--since", "1000d",
                "--format", "md", "--topk", "5"]))
            @test r.code == 0
            @test occursin("# LogClustering insights report", r.out)
            @test occursin("## Top rules", r.out)
            @test occursin("## Novel templates", r.out)
            @test occursin("## Top cluster transitions", r.out)
            @test occursin("## Episodes", r.out)
            @test occursin("## Pinned-pattern activity", r.out)
        end
    end

    @testset "report --format json emits a single JSON dict" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            r = _capture(() -> CLI.main(["report",
                "--memory", db_path, "--since", "1000d",
                "--format", "json"]))
            @test r.code == 0
            d = JSON3.read(r.out)
            for key in ("top_rules", "novel", "transitions",
                        "episodes", "pinned")
                @test haskey(d, key)
            end
        end
    end

    @testset "report --format html is self-contained with every section" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            out = joinpath(dir, "report.html")
            r = _capture(() -> CLI.main(["report",
                "--memory", db_path, "--since", "1000d",
                "--format", "html", "--out", out, "--topk", "5"]))
            @test r.code == 0
            html = read(out, String)
            # Well-formed, self-contained page.
            @test occursin("<!doctype html>", html)
            @test occursin("<style>", html) && occursin("</html>", html)
            # No external requests: no src=/href= to a network resource.
            @test !occursin("http://", html) && !occursin("https://", html)
            # Every section header renders.
            for h in ("Trigger timeline", "Top rules", "Novel templates",
                      "Top cluster transitions", "Episodes",
                      "Pinned-pattern activity", "Embedding scatter")
                @test occursin(h, html)
            end
            # The timeline bar chart and a data row from the fixture.
            @test occursin("class=\"timeline\"", html)
            @test occursin("<td>a</td>", html)          # rule "a" from fixture
            # No embeddings persisted → scatter degrades to a note, not an SVG.
            @test occursin("no embeddings persisted", html)
            @test !occursin("<svg", html)
        end
    end

    @testset "report rejects an unknown --format" begin
        mktempdir() do dir
            db_path = joinpath(dir, "t.sqlite")
            _fill_fixture(db_path)
            r = _capture(() -> CLI.main(["report",
                "--memory", db_path, "--since", "1000d", "--format", "pdf"]))
            @test r.code == 2                       # ArgumentError → exit 2
            @test occursin("unknown --format", r.err)
        end
    end

end
