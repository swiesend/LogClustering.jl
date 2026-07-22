using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Memory.SQLite
using LogClustering.Memory.SQLite: open_db, migrate!, triggers, top_rules,
                                    cluster_id_sequence, epoch_ms_since
using LogClustering.StructuredLog
using JSON3

const _MEM_E2E_LOG = IOBuffer()
StructuredLog.set_format!(:json; stream = _MEM_E2E_LOG)

function _e2e_capture(f)
    orig_out, orig_err = stdout, stderr
    rr_out = Pipe(); Base.link_pipe!(rr_out, reader_supports_async = true,
                                     writer_supports_async = true)
    rr_err = Pipe(); Base.link_pipe!(rr_err, reader_supports_async = true,
                                     writer_supports_async = true)
    redirect_stdout(rr_out)
    redirect_stderr(rr_err)
    code = try
        f()
    finally
        redirect_stdout(orig_out); redirect_stderr(orig_err)
        close(rr_out.in); close(rr_err.in)
    end
    return (code = code, out = read(rr_out, String), err = read(rr_err, String))
end

function _e2e_stdin(path, f)
    orig = stdin
    open(path, "r") do io
        redirect_stdin(io)
        try; f(); finally; redirect_stdin(orig); end
    end
end

function _write_lines(lines::Vector{String})
    path = tempname() * ".log"
    open(path, "w") do io
        for l in lines
            println(io, l)
        end
    end
    return path
end

function _write_rules(d)
    path = tempname() * ".json"
    open(path, "w") do io
        JSON3.write(io, d)
    end
    return path
end

@testset "Memory — stream --memory persistence" begin

    @testset "triggers land in SQLite + session is finalised" begin
        rules_path = _write_rules(Dict(
            "version" => 1,
            "defaults" => Dict("warmup_required" => false, "cooldown_s" => 0),
            "rules" => [Dict("id" => "fatal", "kind" => "keyword",
                              "field" => "line",
                              "keywords" => ["FATAL", "OOM"],
                              "case_sensitive" => false,
                              "severity" => "crit")]))
        data_path = _write_lines([
            "INFO startup",
            "kernel: Out of memory: OOM-killer fired",
            "INFO ready",
            "FATAL: subsystem failed",
            "INFO heartbeat",
        ])
        db_path = tempname() * ".sqlite"

        r = _e2e_stdin(data_path, () -> _e2e_capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--memory", db_path,
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))
        @test r.code == 0

        # Re-open the DB and assert state landed.
        db = open_db(db_path; create = false)
        rows = triggers(db; limit = 100)
        @test length(rows) == 2
        @test all(r -> String(r.rule_id) == "fatal", rows)

        # Session row finalised with exit_code = 0.
        sess_rows = LogClustering.Memory.SQLite._rows(db,
            "SELECT id, ended_at, exit_code FROM sessions")
        @test length(sess_rows) == 1
        @test sess_rows[1].ended_at !== missing
        @test Int(sess_rows[1].exit_code) == 0
    end

    @testset "--persist-lines sampled stores every Nth line" begin
        rules_path = _write_rules(Dict(
            "version" => 1,
            "defaults" => Dict("warmup_required" => false, "cooldown_s" => 0),
            "rules" => []))
        data_path = _write_lines(["line $i" for i in 1:20])
        db_path = tempname() * ".sqlite"

        _e2e_stdin(data_path, () -> _e2e_capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--memory", db_path,
                      "--persist-lines", "sampled",
                      "--persist-lines-rate", "5",
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))

        db = open_db(db_path; create = false)
        line_rows = LogClustering.Memory.SQLite._rows(db,
            "SELECT COUNT(*) AS n FROM lines")
        # With rate=5 and 20 lines, we keep lines 1, 6, 11, 16 = 4 records.
        @test Int(line_rows[1].n) == 4
    end

    @testset "two --persist-lines sessions against one DB don't collide" begin
        rules_path = _write_rules(Dict("version" => 1, "rules" => []))
        db_path = tempname() * ".sqlite"
        for run in 1:2
            data_path = _write_lines(["run$run line $i" for i in 1:5])
            r = _e2e_stdin(data_path, () -> _e2e_capture(() ->
                CLI.main(["stream", "--rules", rules_path,
                          "--memory", db_path,
                          "--persist-lines", "all",
                          "--warmup-lines", "0", "--warmup-seconds", "0",
                          "--status-interval", "0",
                          "--quiet", "--log-level", "error"])))
            @test r.code == 0
        end
        db = open_db(db_path; create = false)
        rows = LogClustering.Memory.SQLite._rows(db,
            "SELECT COUNT(*) AS n FROM lines")
        # Both sessions' 5 lines each — the pre-v2 PK collision would
        # have dropped the whole second batch (and killed the writer).
        @test Int(rows[1].n) == 10
    end

    @testset "--persist-lines-rate 1 keeps every line" begin
        rules_path = _write_rules(Dict("version" => 1, "rules" => []))
        data_path = _write_lines(["l$i" for i in 1:7])
        db_path = tempname() * ".sqlite"
        _e2e_stdin(data_path, () -> _e2e_capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--memory", db_path,
                      "--persist-lines", "sampled",
                      "--persist-lines-rate", "1",
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))
        db = open_db(db_path; create = false)
        rows = LogClustering.Memory.SQLite._rows(db,
            "SELECT COUNT(*) AS n FROM lines")
        @test Int(rows[1].n) == 7
    end

    @testset "--memory survives no triggers (empty triggers table)" begin
        rules_path = _write_rules(Dict("version" => 1, "rules" => []))
        data_path = _write_lines(["a", "b", "c"])
        db_path = tempname() * ".sqlite"

        r = _e2e_stdin(data_path, () -> _e2e_capture(() ->
            CLI.main(["stream", "--rules", rules_path,
                      "--memory", db_path,
                      "--warmup-lines", "0", "--warmup-seconds", "0",
                      "--status-interval", "0",
                      "--quiet", "--log-level", "error"])))
        @test r.code == 0

        db = open_db(db_path; create = false)
        rows = triggers(db; limit = 10)
        @test isempty(rows)
        sess_rows = LogClustering.Memory.SQLite._rows(db,
            "SELECT id, exit_code FROM sessions")
        @test length(sess_rows) == 1
        @test Int(sess_rows[1].exit_code) == 0
    end

end
