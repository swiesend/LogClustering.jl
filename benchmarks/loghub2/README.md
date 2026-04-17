# LogHub-2.0 benchmark harness

Plan 001 Stage F uses [LogHub-2.0][loghub] (Zhu et al. ISSRE 2023,
14 datasets, ~50 k annotated lines each) as the single source of
ground truth for every parser / clusterer in the repo.

[loghub]: https://github.com/logpai/logparser

## Fetch the 2k subsets

The bench ships on the 2k-line annotated subsets (enough for CI,
< 0.2 MB each) published alongside the logparser repo:

```sh
# Default: HDFS, Apache, OpenSSH — enough to run `run.jl` end-to-end.
julia --project benchmarks/loghub2/download.jl

# Full set of 14 datasets referenced in the plan.
julia --project benchmarks/loghub2/download.jl --all

# Specific subset.
julia --project benchmarks/loghub2/download.jl HDFS Spark
```

Downloads land under `benchmarks/loghub2/data/<Dataset>/`; that
directory is `.gitignore`d.

## One dataset, one parser

```sh
julia --project benchmarks/loghub2/run.jl path/to/HDFS_2k.log_structured.csv num_mask
```

prints one TSV row:

```
dataset          parser              n    PA%    GA%   FGA%   FTA%   NMI%   ARI%   Pur%     V%       wall
HDFS_2k          num_mask         2000  16.30  94.10  92.05  54.45  97.21  95.80  97.60  96.14   0.012s
```

Set `LC_HEADER=1` on the first invocation to also emit a column header.

## All datasets, all parsers

```sh
mkdir -p out
LC_HEADER=1 julia --project benchmarks/loghub2/run.jl data/HDFS_2k.log_structured.csv identity >  out/table.tsv
for ds in data/*_structured.csv; do
  for p in identity constant num_mask; do
    julia --project benchmarks/loghub2/run.jl "$ds" "$p" >> out/table.tsv
  done
done
```

## Metrics

- **PA** — fraction of lines whose inferred template string exactly
  equals the ground-truth template string.
- **GA** — fraction of lines whose inferred group (set of sibling
  lines) exactly equals the ground-truth group.
- **FGA / FTA** — grouping F1 (pair-based) and template F1
  (set-match-based); the 2023-era LogHub-2.0 metrics.
- **NMI / ARI / Purity / V-measure** — standard clustering metrics.

Implementations live in `src/Eval/Metrics.jl` (pure Julia, dependency-free).

## Adding a parser

Edit `benchmarks/loghub2/run.jl` and add an entry to `PARSERS`:

```julia
PARSERS["drain"] = function(lines)
    # …
    return inferred_templates   # Vector{String}, one per input line
end
```

Your function must return one template string per input line, in order.
`run_parser` scores it against the ground truth and prints the
per-dataset metrics row.

## Current baselines (sanity only)

| parser      | notes |
|-------------|-------|
| `identity`  | one template per line — floor for PA (0 % on any non-trivial dataset) |
| `constant`  | one global template (`<*>`) — floor for template count |
| `num_mask`  | `\\d+ → <NUM>` then literal-string grouping — first non-trivial baseline |

The plan's full baseline table (Drain, Brain, LogPPT, LILAC, BGE+HDBSCAN,
KATE-original, KATE-modernised, VQ-VAE, SimCSE-logs) lands in follow-up
PRs as each parser gets a concrete implementation or Python binding.
