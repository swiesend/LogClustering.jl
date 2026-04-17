#!/usr/bin/env -S julia --project
#
# Run every parser in `run.jl` across every downloaded LogHub-2.0
# dataset, emit a Markdown baseline table (and TSV) to stdout. Assumes
# `download.jl` has already populated benchmarks/loghub2/data/.
#
#   julia --project benchmarks/loghub2/sweep.jl

include(joinpath(@__DIR__, "run.jl"))

using Printf

const DATA_DIR = joinpath(@__DIR__, "data")
const PARSERS_ORDER = ["identity", "constant", "num_mask", "mask", "drain", "mask+drain"]

function datasets_found()
    out = String[]
    isdir(DATA_DIR) || return out
    for d in sort(readdir(DATA_DIR))
        path = joinpath(DATA_DIR, d, "$(d)_2k.log_structured.csv")
        isfile(path) && push!(out, d)
    end
    return out
end

function main()
    dss = datasets_found()
    if isempty(dss)
        println(stderr, "No datasets under $DATA_DIR — run download.jl first.")
        exit(1)
    end

    rows = NamedTuple[]
    for ds in dss
        path = joinpath(DATA_DIR, ds, "$(ds)_2k.log_structured.csv")
        dataset = load_loghub(path; name = ds)
        for p in PARSERS_ORDER
            fn = PARSERS[p]
            report = run_parser(dataset, fn; parser_name = p)
            push!(rows, (
                dataset = ds,
                parser  = p,
                n       = report.n,
                pa      = 100 * report.pa,
                ga      = 100 * report.ga,
                fga     = 100 * report.fga[3],
                fta     = 100 * report.fta[3],
                nmi     = 100 * report.nmi,
                ari     = 100 * report.ari,
                purity  = 100 * report.purity,
                v       = 100 * report.v,
                wall    = report.elapsed_s,
            ))
        end
    end

    # Markdown table.
    println("# LogHub-2.0 baselines (2k subsets)")
    println()
    println("| dataset | parser | n | PA% | GA% | FGA% | FTA% | NMI% | ARI% | Pur% | V% | wall (s) |")
    println("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows
        @printf(
            "| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f |\n",
            r.dataset, r.parser, r.n, r.pa, r.ga, r.fga, r.fta,
            r.nmi, r.ari, r.purity, r.v, r.wall,
        )
    end

    # Per-dataset best parser by NMI.
    println()
    println("## Best NMI per dataset")
    println()
    println("| dataset | best parser | NMI% |")
    println("|---|---|---:|")
    by_ds = Dict{String, NamedTuple}()
    for r in rows
        prev = get(by_ds, r.dataset, nothing)
        if prev === nothing || r.nmi > prev.nmi
            by_ds[r.dataset] = r
        end
    end
    for ds in dss
        r = by_ds[ds]
        @printf("| %s | %s | %.2f |\n", r.dataset, r.parser, r.nmi)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
