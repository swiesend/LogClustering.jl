#!/usr/bin/env -S julia --project
#
# LogHub-2.0 benchmark harness — plan 001 Stage F entry point.
#
# Usage:
#
#   julia --project benchmarks/loghub2/run.jl <dataset.csv> [parser]
#
# `dataset.csv` is any `*_structured.csv` from the LogHub-2.0 release
# (https://github.com/logpai/logparser). `parser` selects one of the
# built-in parsers (see `PARSERS` below); default `identity` treats
# every raw log line as its own template — the uninformative floor
# that every real parser must beat.
#
# Results are printed as a single TSV row; append with `>> results.tsv`
# to build the plan's baseline table.

using LogClustering
using LogClustering.Harness: load_loghub, run_parser, format_report, Dataset
using LogClustering.Rust: infer_regex

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

"Trivial baseline: every raw line is a unique template (floor for PA)."
parse_identity(lines) = [String(l) for l in lines]

"Even more trivial: one global template (floor for template-count)."
parse_constant(lines) = fill("<*>", length(lines))

"""
Regex-cluster baseline: digit runs become `<NUM>`, then identical
literal strings form the template. This is a step between `identity`
and proper parsers like Drain — useful to sanity-check the metrics.
"""
function parse_num_mask(lines)
    re = r"\b\d+\b"
    return [replace(String(l), re => "<NUM>") for l in lines]
end

const PARSERS = Dict{String, Function}(
    "identity"  => parse_identity,
    "constant"  => parse_constant,
    "num_mask"  => parse_num_mask,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main(args)
    if isempty(args) || args[1] in ("-h", "--help")
        println("""
        usage: run.jl <path-to-*_structured.csv> [parser]

        Available parsers: $(join(sort(collect(keys(PARSERS))), ", "))
        """)
        return
    end
    path = args[1]
    parser_name = length(args) >= 2 ? args[2] : "identity"
    parser_fn = get(PARSERS, parser_name) do
        error("unknown parser: $parser_name. Known: $(collect(keys(PARSERS)))")
    end

    dataset = load_loghub(path)
    report = run_parser(dataset, parser_fn; parser_name = parser_name)

    # Header on first call (when appending to a fresh file, set $LC_HEADER=1).
    if get(ENV, "LC_HEADER", "0") == "1"
        println(rpad("dataset", 16), "  ",
                rpad("parser", 16), "  ",
                lpad("n", 6), "  ",
                lpad("PA%", 6), "  ", lpad("GA%", 6), "  ",
                lpad("FGA%", 6), "  ", lpad("FTA%", 6), "  ",
                lpad("NMI%", 6), "  ", lpad("ARI%", 6), "  ",
                lpad("Pur%", 6), "  ", lpad("V%", 6), "  ",
                lpad("wall", 9))
    end
    println(format_report(report))
end

isinteractive() || main(ARGS)
