#
# Documenter.jl build script. Run locally:
#
#   julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia --project=docs docs/make.jl
#
# CI uses the same invocation with `deploy_docs` enabled by a
# GitHub Pages secret; disabled here so a plain `make.jl` run only
# builds static HTML under `docs/build/`.

using Documenter
using LogClustering
using LogClustering: KATE, DeepKATE, Framing, Masking, Dedup,
                     Episodes, Instance, SeqLSTM, Drain3,
                     Metrics, Compression, Harness, CV,
                     Sparsity, Pipeline, Rust

makedocs(
    sitename = "LogClustering.jl",
    authors  = "Sebastian Wiesendahl and contributors",
    remotes  = nothing,                 # offline build; no edit/source links
    modules  = [
        LogClustering,
        KATE, DeepKATE,
        Framing, Masking, Dedup,
        Episodes, Instance, SeqLSTM, Drain3,
        Metrics, Compression, Harness, CV,
        Sparsity, Pipeline,
        Rust,
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link  = nothing,
        canonical  = "https://swiesend.github.io/LogClustering.jl/",
    ),
    pages = [
        "Home"              => "index.md",
        "Vision"            => "vision.md",
        "Plan 001"          => "plan-001.md",
        "Pipeline" => [
            "Pre-processing" => "preproc.md",
            "Parsers"        => "parsers.md",
            "Models"         => "models.md",
            "Clustering"     => "cluster.md",
            "Mining"         => "mining.md",
            "Anomaly"        => "anomaly.md",
            "RCA"            => "rca.md",
        ],
        "Evaluation" => [
            "Metrics"        => "metrics.md",
            "Compression"    => "compression.md",
            "Harness + CV"   => "harness.md",
        ],
        "FFI"               => "rust.md",
        "Benchmarks"        => "benchmarks.md",
    ],
    checkdocs = :none,
    warnonly  = true,
)
