using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Config
using LogClustering.TUI
using LogClustering.Memory.SQLite: open_db, migrate!, insert_session!,
                                    insert_trigger!
using LogClustering.StructuredLog
using JSON3
using Dates: Dates, DateTime

const _TUI_LOG = IOBuffer()
StructuredLog.set_format!(:json; stream = _TUI_LOG)

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

function _e2e_run(data_path, f)
    orig = stdin
    open(data_path, "r") do io
        redirect_stdin(io)
        try; f(); finally; redirect_stdin(orig); end
    end
end

function _seeded_db(path)
    db = open_db(path)
    migrate!(db)
    sid = insert_session!(db; host = "h")
    base = DateTime("2026-04-27T14:00:00")
    for i in 1:3
        insert_trigger!(db; session_id = sid, rule_id = "fatal",
                        rule_kind = "keyword", severity = "crit",
                        line_id = i, line = "boom $i",
                        fields = Dict("matched" => ["FATAL"]),
                        ts = base + Dates.Second(i),
                        drain_cluster_id = 1)
    end
    return db
end

@testset "TUI.render" begin

    @testset "colour=false strips ANSI escapes" begin
        mktempdir() do dir
            path = joinpath(dir, "t.sqlite")
            db = _seeded_db(path)
            s = TUI.render(db; since = "1000d", colour = false)
            @test occursin("Top rules", s)
            @test occursin("fatal", s)
            @test occursin("boom 1", s) || occursin("boom 3", s)
            # No ANSI CSI sequences.
            @test !occursin("\e[", s)
        end
    end

    @testset "renders an empty-window placeholder" begin
        mktempdir() do dir
            path = joinpath(dir, "empty.sqlite")
            db = open_db(path); migrate!(db)
            s = TUI.render(db; since = "1h", colour = false)
            @test occursin("no triggers in window", s)
        end
    end

    @testset "ANSI mode emits escape sequences" begin
        mktempdir() do dir
            path = joinpath(dir, "t.sqlite")
            db = _seeded_db(path)
            s = TUI.render(db; since = "1000d", colour = true)
            @test occursin("\e[", s)
        end
    end

end

@testset "CLI — top --once renders one frame and exits" begin
    mktempdir() do dir
        path = joinpath(dir, "t.sqlite")
        _seeded_db(path)
        r = _capture(() -> CLI.main(["top", "--memory", path,
                                      "--since", "1000d", "--once"]))
        @test r.code == 0
        @test occursin("Top rules", r.out)
    end
end

@testset "Config — load + merge" begin

    @testset "default_path honours env vars" begin
        # Save / restore the env so other tests aren't affected.
        old_cfg = get(ENV, "LOGCLUSTERING_CONFIG", nothing)
        ENV["LOGCLUSTERING_CONFIG"] = "/tmp/xyz.toml"
        try
            @test Config.default_path() == "/tmp/xyz.toml"
        finally
            old_cfg === nothing ? delete!(ENV, "LOGCLUSTERING_CONFIG") :
                                    (ENV["LOGCLUSTERING_CONFIG"] = old_cfg)
        end
    end

    @testset "load returns empty Dict when file missing" begin
        @test Config.load("/no/such/path") == Dict{String, Any}()
    end

    @testset "merge_into! respects cli-provided + known keys" begin
        mktempdir() do dir
            path = joinpath(dir, "x.toml")
            open(path, "w") do io
                write(io, """
                [stream]
                rules = "/etc/r.json"
                log-level = "warn"
                unknown-future-key = "ignored"
                """)
            end
            cfg = Config.load(path)
            opts = Dict{String, Any}("rules" => "", "log-level" => "info")
            # Operator passed --rules on CLI; we shouldn't overwrite it.
            Config.merge_into!(opts, cfg["stream"];
                cli_provided = Set(["rules"]))
            @test opts["rules"] == ""
            @test opts["log-level"] == "warn"
            @test !haskey(opts, "unknown-future-key")
        end
    end

end

@testset "CLI — init bootstraps + doctor verifies" begin
    mktempdir() do dir
        prefix = joinpath(dir, "cfg")
        # Point XDG_DATA_HOME at a temp dir too so the db doesn't
        # land in ~/.local.
        old_data = get(ENV, "XDG_DATA_HOME", nothing)
        ENV["XDG_DATA_HOME"] = joinpath(dir, "data")
        try
            r = _capture(() -> CLI.main(["init", "--prefix", prefix]))
            @test r.code == 0
            d = JSON3.read(r.out)
            @test isfile(String(d["config"]))
            @test isfile(String(d["rules"]))
            @test isfile(String(d["db"]))

            r2 = _capture(() -> CLI.main(["doctor",
                "--config", String(d["config"]),
                "--memory", String(d["db"]),
                "--rules",  String(d["rules"]),
                "--json"]))
            @test r2.code == 0
            findings = JSON3.read(r2.out)
            statuses = [String(f["status"]) for f in findings]
            @test all(s -> s in ("ok", "warn"), statuses)
            # `config`, `memory`, `rules` all OK on a fresh init.
            ok_labels = [String(f["label"]) for f in findings
                         if String(f["status"]) == "ok"]
            @test "memory" in ok_labels
            @test "rules"  in ok_labels
        finally
            old_data === nothing ? delete!(ENV, "XDG_DATA_HOME") :
                                    (ENV["XDG_DATA_HOME"] = old_data)
        end
    end
end

@testset "config file wires into stream (CLI > config > default)" begin
    mktempdir() do dir
        db_path  = joinpath(dir, "cfg.sqlite")
        cfg_path = joinpath(dir, "config.toml")
        open(cfg_path, "w") do io
            write(io, """
            [stream]
            memory = "$db_path"
            log-level = "error"
            """)
        end
        rules_path = joinpath(dir, "rules.json")
        open(rules_path, "w") do io
            JSON3.write(io, Dict("version" => 1, "rules" => []))
        end
        data_path = joinpath(dir, "in.log")
        open(data_path, "w") do io; println(io, "a"); println(io, "b"); end

        old_cfg = get(ENV, "LOGCLUSTERING_CONFIG", nothing)
        ENV["LOGCLUSTERING_CONFIG"] = cfg_path
        try
            # --memory is NOT passed on the CLI: it must come from config.
            r = _e2e_run(data_path, () -> _capture(() ->
                CLI.main(["stream", "--rules", rules_path,
                          "--warmup-lines", "0", "--warmup-seconds", "0",
                          "--status-interval", "0", "--quiet"])))
            @test r.code == 0
            # The config's memory path was used → the db exists + migrated.
            @test isfile(db_path)
            db = LogClustering.Memory.SQLite.open_db(db_path; create = false)
            srows = LogClustering.Memory.SQLite._rows(db,
                "SELECT COUNT(*) AS n FROM sessions")
            @test Int(srows[1].n) == 1
        finally
            old_cfg === nothing ? delete!(ENV, "LOGCLUSTERING_CONFIG") :
                                   (ENV["LOGCLUSTERING_CONFIG"] = old_cfg)
        end
    end
end

@testset "TUI truncates a multibyte line without StringIndexError" begin
    mktempdir() do dir
        path = joinpath(dir, "t.sqlite")
        db = open_db(path); migrate!(db)
        sid = insert_session!(db; host = "h")
        # A line with a multi-byte char (…) straddling byte 80.
        line = repeat("x", 78) * "…" * repeat("y", 40)
        insert_trigger!(db; session_id = sid, rule_id = "r",
                        rule_kind = "keyword", severity = "warn",
                        line_id = 1, line = line, fields = Dict(),
                        drain_cluster_id = 1)
        # Must render without throwing.
        s = TUI.render(db; since = "1000d", colour = false)
        @test occursin("Latest triggers", s)
    end
end
