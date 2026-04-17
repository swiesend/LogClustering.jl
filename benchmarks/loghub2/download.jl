#!/usr/bin/env -S julia --project
#
# Fetch LogHub-2.0 2k-line annotated subsets.
#
#   julia --project benchmarks/loghub2/download.jl              # default set (HDFS, Apache, OpenSSH)
#   julia --project benchmarks/loghub2/download.jl --all        # all 14 standard datasets
#   julia --project benchmarks/loghub2/download.jl HDFS Spark   # a custom subset
#
# Files land in `benchmarks/loghub2/data/<Dataset>/<Dataset>_2k.log_structured.csv`
# and are ready to hand to `run.jl`. `benchmarks/loghub2/data/` is gitignored.
#
# Source: https://github.com/logpai/logparser (raw.githubusercontent.com).
# The 2k subsets are small enough (~0.2 MB each) to be safe for CI.

using Downloads: download

const REPO_BASE = "https://raw.githubusercontent.com/logpai/logparser/main/data/loghub_2k"

"The 14 LogHub datasets referenced in plan 001 Stage F."
const ALL_DATASETS = [
    "Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp",
    "HPC", "Linux", "Mac", "OpenSSH", "OpenStack", "Proxifier",
    "Spark", "Thunderbird", "Windows", "Zookeeper",
]

"Small default set — enough to run the bench harness end-to-end on CI."
const DEFAULT_DATASETS = ["HDFS", "Apache", "OpenSSH"]

struct Artifact
    name::String
    relative_path::String
end

function artifacts_for(dataset::AbstractString)
    [Artifact(dataset, "$(dataset)_2k.log_structured.csv")]
end

function dest_dir()
    joinpath(@__DIR__, "data")
end

function fetch_one(dataset::AbstractString; overwrite::Bool = false)
    out_dir = joinpath(dest_dir(), dataset)
    isdir(out_dir) || mkpath(out_dir)
    any_fetched = false
    for art in artifacts_for(dataset)
        url = "$(REPO_BASE)/$(dataset)/$(art.relative_path)"
        out = joinpath(out_dir, art.relative_path)
        if isfile(out) && !overwrite
            println("  skip   ", relpath(out, @__DIR__), "  (already present)")
            continue
        end
        print("  fetch  ", rpad(dataset, 14), "  ← ", url, " ... ")
        try
            download(url, out)
            println("ok  (", round(stat(out).size / 1024; digits = 1), " KB)")
            any_fetched = true
        catch err
            println("FAIL")
            @warn "download failed" dataset = dataset url = url err = err
            rethrow(err)
        end
    end
    return any_fetched
end

function parse_args(args)
    datasets = String[]
    overwrite = false
    use_all = false
    for a in args
        if a in ("-h", "--help")
            println("""
            usage: download.jl [--all | --force | <Dataset>...]

            Flags:
              --all     Fetch all $(length(ALL_DATASETS)) LogHub-2.0 datasets.
              --force   Re-fetch even when a local copy exists.
              -h,--help Show this help.

            Datasets: $(join(ALL_DATASETS, ", "))
            Default:  $(join(DEFAULT_DATASETS, ", "))
            """)
            return nothing
        elseif a == "--all"
            use_all = true
        elseif a == "--force"
            overwrite = true
        elseif startswith(a, "-")
            error("unknown flag: $a")
        else
            if !(a in ALL_DATASETS)
                error("unknown dataset: $a. Known: " * join(ALL_DATASETS, ", "))
            end
            push!(datasets, a)
        end
    end
    list = use_all ? ALL_DATASETS :
           isempty(datasets) ? DEFAULT_DATASETS :
           datasets
    return (list = list, overwrite = overwrite)
end

function main(args)
    cfg = parse_args(args)
    cfg === nothing && return
    println("LogHub-2.0 download → ", relpath(dest_dir(), pwd()), "/")
    println("  targets: ", join(cfg.list, ", "))
    for ds in cfg.list
        fetch_one(ds; overwrite = cfg.overwrite)
    end
    println()
    println("Done. Try:")
    example = joinpath("benchmarks", "loghub2", "data",
                       first(cfg.list),
                       "$(first(cfg.list))_2k.log_structured.csv")
    println("  julia --project benchmarks/loghub2/run.jl $(example) num_mask")
end

isinteractive() || main(ARGS)
