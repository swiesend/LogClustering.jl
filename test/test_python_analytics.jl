# Integration tests for the Python-backed analytics tier (O1 UMAP/HDBSCAN
# in RCA, O2 embedding scatter). These run for real ONLY when the uv
# virtualenv under `py/.venv` is present; otherwise every case is a
# `@test_skip` so CI without Python still passes. The suite-wide env
# wiring lives at the top of runtests.jl; we repeat it here (via `get!`,
# so it never overrides) so this file also runs standalone under jlrun.

using Test
using LogClustering
using LogClustering.CLI
using LogClustering.Pipeline: umap_reduce, hdbscan_cluster, umap_hdbscan
using LogClustering.Memory.SQLite: open_db, migrate!, insert_session!, insert_line!
using LogClustering.Memory.Insights: embedding_scatter
using Random: MersenneTwister
using Dates: DateTime

function _py_venv_python()
    venv = abspath(joinpath(@__DIR__, "..", "py", ".venv", "bin", "python"))
    return isfile(venv) ? venv : nothing
end

let py = _py_venv_python()
    if py !== nothing
        get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")
        get!(ENV, "JULIA_PYTHONCALL_EXE", py)
    end
end

const _PY_AVAILABLE = _py_venv_python() !== nothing

@testset "Python analytics (UMAP/HDBSCAN)" begin
    if !_PY_AVAILABLE
        @info "py/.venv absent — skipping live UMAP/HDBSCAN tests"
        @test_skip false
    else
        @testset "umap_reduce / hdbscan / umap_hdbscan work in one call frame" begin
            # All inside a function: exercises the world-age path that the
            # `invokelatest` boundary in Pipeline.jl fixes (a bare
            # first-use inside a single frame used to throw a MethodError).
            function run_all()
                rng = MersenneTwister(1)
                X = hcat(randn(rng, 6, 30) .- 3.0, randn(rng, 6, 30) .+ 3.0)
                Y = umap_reduce(X; n_neighbors = 10, n_components = 2,
                                random_state = 42)
                r = umap_hdbscan(X; l2 = false, n_neighbors = 10,
                                 min_cluster_size = 5)
                return (Y = Y, assign = r.assignments, emb = r.embedding)
            end
            out = run_all()
            @test size(out.Y) == (2, 60)
            @test all(isfinite, out.Y)
            @test length(out.assign) == 60
            @test length(unique(out.assign)) >= 2      # two blobs separated
            @test size(out.emb, 2) == 60
        end

        @testset "rca --cluster hdbscan runs end-to-end via the CLI" begin
            dir = mktempdir()
            data = joinpath(dir, "d.log")
            open(data, "w") do io
                for i in 1:60
                    println(io, i % 2 == 0 ? "INFO request $i ok" :
                                "INFO user $i logged in")
                end
                for _ in 1:6; println(io, "ERROR db connection lost"); end
            end
            model = joinpath(dir, "k.jld2")
            @test CLI.cmd_train(String["--kind", "deep_kate", "--data", data,
                "--out", model, "--epochs", "2", "--quiet"]) == 0
            out = joinpath(dir, "rca.md")
            @test CLI.cmd_rca(String["--model", model, "--data", data,
                "--out", out, "--embedder", "model", "--cluster", "hdbscan",
                "--min-cluster-size", "4", "--min-sup", "3", "--max-gap", "3",
                "--max-dur", "5"]) == 0
            @test occursin("RCA report", read(out, String))
        end

        @testset "embedding_scatter projects persisted vectors to 2-D" begin
            dir = mktempdir()
            db = open_db(joinpath(dir, "m.sqlite")); migrate!(db)
            sid = insert_session!(db; host = "h"); base = DateTime(2026, 1, 1)
            for i in 1:40
                insert_line!(db; session_id = sid, line_id = i, line = "L$i",
                    ts = base, drain_cluster_id = (i % 3) + 1,
                    embedding = Float32[sin(i), cos(i), Float32(i % 5),
                                        Float32(i % 7)])
            end
            rows = embedding_scatter(db; since = 0, n_neighbors = 10,
                                     random_state = 42)
            @test length(rows) == 40
            @test all(r -> isfinite(r.x) && isfinite(r.y), rows)
            @test all(r -> r.cluster in (1, 2, 3), rows)
        end

        @testset "report --format html renders a populated SVG scatter" begin
            dir = mktempdir()
            db_path = joinpath(dir, "m.sqlite")
            db = open_db(db_path); migrate!(db)
            sid = insert_session!(db; host = "h"); base = DateTime(2026, 1, 1)
            for i in 1:40
                insert_line!(db; session_id = sid, line_id = i, line = "L$i",
                    ts = base, drain_cluster_id = (i % 3) + 1,
                    embedding = Float32[sin(i), cos(i), Float32(i % 4)])
            end
            out = joinpath(dir, "report.html")
            @test CLI.cmd_report(String["--memory", db_path, "--since", "1000d",
                "--format", "html", "--out", out]) == 0
            html = read(out, String)
            @test occursin("<svg", html)               # scatter is populated
            @test occursin("<circle", html)            # one dot per embedded line
            @test !occursin("no embeddings persisted", html)
        end
    end
end
