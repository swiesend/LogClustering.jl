#!/usr/bin/env julia
# ---------------------------------------------------------------------------
# stream_perf.jl — throughput smoke for the streaming hot path.
#
#   julia --project benchmarks/stream_perf.jl [N_LINES]
#
# Generates a synthetic corpus and times `logcluster stream` under a few
# configurations, reporting lines/s. This is a *smoke* (a rough, machine-
# dependent number to catch regressions and show the effect of the perf
# levers), NOT a hard test — the assertions live in the test suite
# (t-digest allocation guard, dedup exactness, batched-NLL equality).
#
# Levers exercised:
#   - drain-only               (cheap baseline; no transformer forward)
#   - drain + decoder NLL      (the expensive path)
#   - + --dedup masked         (reuse NLL for near-duplicate templates)
#   - + --batch-lines 32       (amortize the transformer forward)
# ---------------------------------------------------------------------------

using LogClustering
using LogClustering.CLI
using Printf

const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000

# A corpus dominated by a handful of recurring templates (like real logs),
# so dedup has something to collapse.
function _gen_corpus(path, n)
    templates = [
        "INFO request GET /api/v1/items took %dms",
        "INFO user %d logged in from 10.0.0.%d",
        "WARN cache miss for key session:%d",
        "INFO worker %d processed job %d",
        "ERROR db connection lost on shard %d",
    ]
    open(path, "w") do io
        for i in 1:n
            t = templates[(i % length(templates)) + 1]
            println(io, replace(t, "%d" => string(i % 1000)))
        end
    end
end

# Time a stream run with its trigger/shutdown JSON silenced, so the
# reported table isn't buried in event output.
function _time_stream(args; n)
    t = redirect_stdout(devnull) do
        @elapsed CLI.cmd_stream(args)
    end
    return n / t
end

function main()
    dir = mktempdir()
    data = joinpath(dir, "corpus.log")
    _gen_corpus(data, N)
    dec = joinpath(dir, "dec.jld2")

    @info "training a small transformer_decoder for the NLL path…"
    CLI.cmd_train(String["--kind", "transformer_decoder", "--data", data,
        "--out", dec, "--seqlen", "16", "--d-model", "64", "--n-layers", "2",
        "--n-heads", "4", "--epochs", "1", "--batch", "64", "--quiet"])

    base = ["--data", data, "--max-events", string(N), "--quiet",
            "--warmup-lines", "0", "--status-interval", "0"]

    configs = [
        ("drain-only",                 String[base...]),
        ("decoder NLL (baseline)",     String[base..., "--model", dec]),
        ("decoder + dedup masked",     String[base..., "--model", dec,
                                              "--dedup", "masked"]),
        ("decoder + dedup + batch-32", String[base..., "--model", dec,
                                              "--dedup", "masked",
                                              "--batch-lines", "32"]),
    ]

    # Warm up (compile the stream path) before timing.
    redirect_stdout(devnull) do
        CLI.cmd_stream(String[base..., "--max-events", "200"])
    end

    println("\n", "="^58)
    @printf("%-30s %12s\n", "config  ($(N) lines)", "lines/s")
    println("-"^58)
    for (name, args) in configs
        rate = _time_stream(args; n = N)
        @printf("%-30s %12.0f\n", name, rate)
    end
    println("="^58)
    println("(smoke only — absolute numbers are machine-dependent)")
end

main()
