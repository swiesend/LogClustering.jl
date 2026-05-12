using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Memory.SQLite: open_db, migrate!, insert_session!,
                                    insert_trigger!
using LogClustering.Memory.PatternCatalog
using LogClustering.Memory.PatternCatalog: Pattern, pin_from_trigger,
                                            pin_manual, list, get_pattern,
                                            enable!, match_line,
                                            to_rules, drain_patterns
# `delete!` clashes with Base.delete!; reach for it explicitly.
const _pc_delete! = LogClustering.Memory.PatternCatalog.delete!
using LogClustering.StructuredLog
using JSON3
using Dates: DateTime

const _PAT_LOG = IOBuffer()
StructuredLog.set_format!(:json; stream = _PAT_LOG)

function _fresh_db()
    path = tempname() * ".sqlite"
    db = open_db(path)
    migrate!(db)
    sid = insert_session!(db; host = "h")
    return (path = path, db = db, sid = sid)
end

@testset "PatternCatalog — author / list / toggle" begin

    @testset "pin_manual + list returns enabled-only by default" begin
        d = _fresh_db()
        pid = pin_manual(d.db; name = "fatal",
            match_kind = :keyword,
            match_keywords = ["FATAL", "OOM"],
            severity = :crit, case_sensitive = false)
        @test pid > 0
        rows = list(d.db)
        @test length(rows) == 1
        @test rows[1].name == "fatal"
        @test rows[1].match_keywords == ["FATAL", "OOM"]

        enable!(d.db, pid, false)
        @test isempty(list(d.db))                 # enabled-only filter
        @test length(list(d.db; enabled_only = false)) == 1
    end

    @testset "delete! removes a row" begin
        d = _fresh_db()
        pid = pin_manual(d.db; name = "rm", match_kind = :keyword,
                          match_keywords = ["x"])
        _pc_delete!(d.db, pid)
        @test get_pattern(d.db, pid) === nothing
    end

    @testset "pin_from_trigger picks up drain_cluster_id" begin
        d = _fresh_db()
        tid = insert_trigger!(d.db; session_id = d.sid,
                              rule_id = "kw", rule_kind = "keyword",
                              severity = "crit", line_id = 1,
                              line = "FATAL panic",
                              fields = Dict("matched" => ["FATAL"]),
                              drain_cluster_id = 42)
        pid = pin_from_trigger(d.db, tid;
            name = "from-trigger-test",
            match_kind = :drain, severity = :crit)
        p = get_pattern(d.db, pid)
        @test p !== nothing
        @test p.match_kind == :drain
        @test p.match_drain_template_id == 42
    end

    @testset "pin_from_trigger with :keyword extracts up to 5 alphanums" begin
        d = _fresh_db()
        tid = insert_trigger!(d.db; session_id = d.sid,
                              rule_id = "r", rule_kind = "keyword",
                              severity = "warn", line_id = 1,
                              line = "ssh login failed for user root from 10.0.0.1",
                              fields = Dict())
        pid = pin_from_trigger(d.db, tid; name = "ssh-fail",
            match_kind = :keyword)
        p = get_pattern(d.db, pid)
        @test 1 <= length(p.match_keywords) <= 5
    end

end

@testset "PatternCatalog — matching" begin

    @testset "match_line dispatches on kind" begin
        kw = Pattern(1, "kw", "", :warn, 60.0, :keyword,
                     nothing, nothing, ["OOM"], false, "", "2026", true)
        @test match_line(kw, Dict("line" => "kernel: OOM-killer fired"))
        @test !match_line(kw, Dict("line" => "all good"))

        rx = Pattern(2, "rx", "", :warn, 60.0, :regex,
                     nothing, Regex("kernel:.*OOM"), String[], false, "", "", true)
        @test match_line(rx, Dict("line" => "kernel: Out of memory: OOM"))
        @test !match_line(rx, Dict("line" => "OOM but not from kernel"))

        dr = Pattern(3, "dr", "", :warn, 60.0, :drain,
                     17, nothing, String[], false, "", "", true)
        @test match_line(dr, Dict("line" => "x",
                                   "drain" => Dict("cluster_id" => 17)))
        @test !match_line(dr, Dict("line" => "x",
                                    "drain" => Dict("cluster_id" => 99)))
        @test !match_line(dr, Dict("line" => "x"))   # no drain block
    end

    @testset "to_rules turns keyword + regex into engine rules" begin
        pats = [
            Pattern(1, "kw", "", :warn, 30.0, :keyword,
                    nothing, nothing, ["OOM"], false, "", "", true),
            Pattern(2, "rx", "", :crit, 5.0, :regex,
                    nothing, Regex("FATAL"), String[], false, "", "", true),
            Pattern(3, "dr", "", :crit, 60.0, :drain,
                    17, nothing, String[], false, "", "", true),
            Pattern(4, "off", "", :warn, 30.0, :keyword,
                    nothing, nothing, ["x"], false, "", "", false),
        ]
        rs = to_rules(pats)
        # Three enabled patterns; only the keyword + regex ones synthesise rules.
        @test length(rs) == 2
        ids = [r.id for r in rs]
        @test "pattern:1:kw" in ids
        @test "pattern:2:rx" in ids
        @test isempty(filter(r -> startswith(r.id, "pattern:3:"), rs))

        @test length(drain_patterns(pats)) == 1
        @test drain_patterns(pats)[1].id == 3
    end

end

@testset "PatternCatalog — stream --use-patterns e2e" begin

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

    function _stdin(path, f)
        orig = stdin
        open(path, "r") do io
            redirect_stdin(io); try; f(); finally; redirect_stdin(orig); end
        end
    end

    @testset "keyword pattern auto-promotes + fires on stream" begin
        d = _fresh_db()
        pid = pin_manual(d.db; name = "fatal_kw",
            match_kind = :keyword, match_keywords = ["BOOM"],
            severity = :crit)

        # Empty rules file so only the synthesised pattern can fire.
        rules_path = tempname() * ".json"
        open(rules_path, "w") do io
            JSON3.write(io, Dict("version" => 1, "rules" => []))
        end
        data_path = tempname() * ".log"
        open(data_path, "w") do io
            println(io, "INFO ok")
            println(io, "BOOM happened")
            println(io, "INFO over")
        end

        r = _stdin(data_path, () -> _capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--use-patterns", d.path,
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))
        @test r.code == 0
        events = [JSON3.read(l) for l in
                  filter(!isempty, split(r.out, '\n'))]
        triggers = filter(e -> String(e["event"]) == "trigger", events)
        @test length(triggers) == 1
        @test String(triggers[1]["rule_id"]) == "pattern:$(pid):fatal_kw"
        @test String(triggers[1]["severity"]) == "crit"
    end

    @testset "drain pattern fires via sidecar" begin
        d = _fresh_db()
        pid = pin_manual(d.db; name = "tpl_42",
            match_kind = :drain, match_drain_template_id = 42,
            severity = :warn)

        rules_path = tempname() * ".json"
        open(rules_path, "w") do io
            JSON3.write(io, Dict("version" => 1, "rules" => []))
        end
        # Drain matching requires a trained drain model + the
        # cluster_id to actually equal 42 on at least one line. Easier
        # to drive the path: build a tiny model on the fly so the
        # stream's _infer pipeline attaches drain info. The Drain
        # parser is deterministic; line "a b c" creates cluster_id 1
        # for the first line. To force a 42 hit we'd need to train on
        # 41 distinct lines — skip that here and instead train a 1-
        # line corpus, then pin pattern against cluster_id 1.
        model_path = tempname() * ".jld2"
        train_data = tempname() * ".log"
        open(train_data, "w") do io
            println(io, "BANG something happened")
        end
        _capture(() -> CLI.main(["train", "--kind", "drain",
                                  "--data", train_data,
                                  "--out", model_path, "--quiet"]))

        # Re-pin against cluster_id 1 (Drain assigns 1 to the first
        # template it ever sees).
        d2 = _fresh_db()
        pid = pin_manual(d2.db; name = "tpl_1",
            match_kind = :drain, match_drain_template_id = 1,
            severity = :warn)

        data_path = tempname() * ".log"
        open(data_path, "w") do io
            println(io, "BANG something happened")
        end

        r = _stdin(data_path, () -> _capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--use-patterns", d2.path,
                      "--model", model_path,
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))
        @test r.code == 0
        events = [JSON3.read(l) for l in
                  filter(!isempty, split(r.out, '\n'))]
        triggers = filter(e -> String(e["event"]) == "trigger", events)
        @test length(triggers) == 1
        @test String(triggers[1]["rule_kind"]) == "drain_pattern"
    end

end
