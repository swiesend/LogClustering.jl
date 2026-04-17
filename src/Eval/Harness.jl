"""
    Eval.Harness

Minimal LogHub-2.0 loader + runner for plan 001 Stage F. Stays
dependency-free by hand-parsing the well-behaved CSV that LogHub
publishes:

```csv
LineId,Content,EventId,EventTemplate
1,INFO: started,E1,INFO: started
...
```

Quoted fields and embedded commas / quotes inside double quotes are
handled; everything else is treated as literal.

Typical usage:

```julia
using LogClustering.Harness
dataset = load_loghub("data/HDFS_2k.log_structured.csv")
# Drop a parser in that takes a vector of raw lines and returns a
# vector of inferred templates:
function my_parser(lines)
    # Trivial "identity" parser.
    return copy(lines)
end
report = run_parser(dataset, my_parser)
```
"""
module Harness

using ..Metrics

export load_loghub, run_parser, Dataset, Report, format_report

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

"One LogHub-2.0 benchmark dataset: raw lines + ground-truth templates."
struct Dataset
    name::String
    lines::Vector{String}
    templates::Vector{String}
    event_ids::Vector{String}
end

Base.length(d::Dataset) = length(d.lines)

function Base.show(io::IO, d::Dataset)
    print(io, "Dataset(", repr(d.name),
          "; n=", length(d.lines),
          ", |templates|=", length(unique(d.templates)), ")")
end

"""
    load_loghub(path::AbstractString; name = basename(path)) -> Dataset

Parse a LogHub-2.0 `*_structured.csv`. Required header columns:
`LineId, Content, EventId, EventTemplate`. Column order does not matter.
"""
function load_loghub(path::AbstractString; name::AbstractString = basename(path))
    header, rows = _read_csv(path)
    idx = Dict{String, Int}(h => i for (i, h) in enumerate(header))
    for k in ("Content", "EventId", "EventTemplate")
        haskey(idx, k) || throw(ArgumentError("missing CSV column: $k"))
    end
    lines     = [row[idx["Content"]]       for row in rows]
    eids      = [row[idx["EventId"]]       for row in rows]
    templates = [row[idx["EventTemplate"]] for row in rows]
    return Dataset(String(name), lines, templates, eids)
end

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

"Metrics for one parser × dataset pair."
struct Report
    dataset::String
    parser::String
    n::Int
    pa::Float64            # parsing accuracy (exact-template match)
    ga::Float64            # group accuracy (multiset-membership match)
    fga::NTuple{3, Float64}  # (precision, recall, F1) — LogHub FGA
    fta::NTuple{3, Float64}  # (precision, recall, F1) — LogHub FTA
    nmi::Float64
    ari::Float64
    purity::Float64
    v::Float64
    elapsed_s::Float64
end

"""
    format_report(r::Report) -> String

Single-row CSV-ish summary suitable for logging or pasting into the
README benchmarks table.
"""
function format_report(r::Report)
    return string(
        rpad(r.dataset, 16), "  ",
        rpad(r.parser, 16), "  ",
        lpad(r.n, 6), "  ",
        lpad(round(100r.pa;     digits = 2), 6), "  ",
        lpad(round(100r.ga;     digits = 2), 6), "  ",
        lpad(round(100r.fga[3]; digits = 2), 6), "  ",
        lpad(round(100r.fta[3]; digits = 2), 6), "  ",
        lpad(round(100r.nmi;    digits = 2), 6), "  ",
        lpad(round(100r.ari;    digits = 2), 6), "  ",
        lpad(round(100r.purity; digits = 2), 6), "  ",
        lpad(round(100r.v;      digits = 2), 6), "  ",
        lpad(round(r.elapsed_s; digits = 3), 8), "s",
    )
end

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

"""
    run_parser(dataset, parser_fn; parser_name = "parser") -> Report

Run `parser_fn(lines) -> Vector{String}` on the dataset's raw `Content`
column and score the returned templates against the ground truth. The
parser must return one inferred template per input line in order.
"""
function run_parser(dataset::Dataset, parser_fn;
                    parser_name::AbstractString = "parser")::Report
    t0 = time()
    pred = parser_fn(dataset.lines)::AbstractVector{<:AbstractString}
    dt = time() - t0
    length(pred) == length(dataset) ||
        throw(DimensionMismatch("parser returned $(length(pred)) templates \
                                 for $(length(dataset)) lines"))
    pa   = Metrics.parsing_accuracy(pred, dataset.templates)
    ga   = Metrics.group_accuracy(pred, dataset.templates)
    fga  = Metrics.grouping_f1(pred, dataset.templates)
    fta  = Metrics.template_group_f1(pred, dataset.templates)
    pred_i = Metrics.to_labels(pred)
    gold_i = Metrics.to_labels(dataset.templates)
    return Report(
        dataset.name, String(parser_name), length(dataset),
        pa, ga, fga, fta,
        Metrics.nmi(pred_i, gold_i),
        Metrics.ari(pred_i, gold_i),
        Metrics.purity(pred_i, gold_i),
        Metrics.v_measure(pred_i, gold_i)[3],
        dt,
    )
end

# ---------------------------------------------------------------------------
# Minimal RFC 4180–ish CSV reader
# ---------------------------------------------------------------------------

function _read_csv(path::AbstractString)
    data = read(path, String)
    # Split into physical lines, but be careful about quoted newlines.
    fields = _split_csv(data)
    isempty(fields) && throw(ArgumentError("empty CSV: $path"))
    header = fields[1]
    rows = fields[2:end]
    expected = length(header)
    for (i, r) in enumerate(rows)
        length(r) == expected ||
            throw(ArgumentError("row $(i + 1) has $(length(r)) fields; expected $expected"))
    end
    return header, rows
end

function _split_csv(s::AbstractString)
    rows = Vector{Vector{String}}()
    row = String[]
    buf = IOBuffer()
    in_quotes = false
    i = firstindex(s)
    n = lastindex(s)
    while i <= n
        ch = s[i]
        if in_quotes
            if ch == '"'
                if i < n && s[nextind(s, i)] == '"'
                    write(buf, '"')
                    i = nextind(s, i)
                else
                    in_quotes = false
                end
            else
                write(buf, ch)
            end
        else
            if ch == ','
                push!(row, String(take!(buf)))
            elseif ch == '\n'
                push!(row, String(take!(buf)))
                push!(rows, row)
                row = String[]
            elseif ch == '\r'
                # swallow; \n (if present) handles row break
            elseif ch == '"'
                in_quotes = true
            else
                write(buf, ch)
            end
        end
        i = nextind(s, i)
    end
    # Flush trailing field / row if any (file missing trailing newline).
    if buf.size > 0 || !isempty(row)
        push!(row, String(take!(buf)))
        push!(rows, row)
    end
    return rows
end

end # module Harness
