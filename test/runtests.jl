using Test
using LogClustering

# When the uv virtualenv under `py/.venv` is present, point PythonCall at
# it before any test loads PythonCall, so the Python-backed analytics
# tests (UMAP/HDBSCAN) run for real instead of skip-guarding. Harmless
# when the venv is absent (the vars stay unset and those tests skip).
let venv = abspath(joinpath(@__DIR__, "..", "py", ".venv", "bin", "python"))
    if isfile(venv)
        get!(ENV, "JULIA_CONDAPKG_BACKEND", "Null")
        get!(ENV, "JULIA_PYTHONCALL_EXE", venv)
    end
end

@testset "LogClustering.jl" begin
    include("test_KATE.jl")
    include("test_deepkate.jl")
    include("test_anomaly.jl")
    include("test_anomaly_sequence.jl")
    include("test_value_novelty.jl")
    include("test_seqlstm.jl")
    include("test_transformer.jl")
    include("test_vqvae.jl")
    include("test_simcse.jl")
    include("test_denoising_ae.jl")
    include("test_softkate.jl")
    include("test_framing.jl")
    include("test_preproc.jl")
    include("test_featurise.jl")
    include("test_episodes.jl")
    include("test_baselines.jl")
    include("test_eval.jl")
    include("test_compression.jl")
    include("test_cv.jl")
    include("test_cluster.jl")
    include("test_python_analytics.jl")
    include("test_drain.jl")
    include("test_template_ids.jl")
    include("test_canonical.jl")
    include("test_postproc.jl")
    include("test_persistence.jl")
    include("test_autotune.jl")
    include("test_rca.jl")
    include("test_tdigest.jl")
    include("test_rrcf.jl")
    include("test_adwin.jl")
    include("test_rules.jl")
    include("test_stream.jl")
    include("test_sinks_webhook.jl")
    include("test_memory_sqlite.jl")
    include("test_memory_redis.jl")
    include("test_pattern_catalog.jl")
    include("test_insights.jl")
    include("test_tui_init_doctor.jl")
    include("test_stream_e2e.jl")
    include("test_memory_e2e.jl")
    include("test_cli.jl")
    include("test_ad.jl")
    include("test_rust.jl")
end
