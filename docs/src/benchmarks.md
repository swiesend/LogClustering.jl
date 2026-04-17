# Benchmarks

Plan 001 Stage F. The LogHub-2.0 harness lives under
[`benchmarks/loghub2/`](https://github.com/swiesend/LogClustering.jl/tree/main/benchmarks/loghub2).

## Fetch the 2 k subsets

```sh
julia --project benchmarks/loghub2/download.jl            # HDFS, Apache, OpenSSH
julia --project benchmarks/loghub2/download.jl --all      # full 16-dataset set
```

## Run one parser against one dataset

```sh
julia --project benchmarks/loghub2/run.jl \
    benchmarks/loghub2/data/HDFS/HDFS_2k.log_structured.csv drain
```

Parser names: `identity`, `constant`, `num_mask`, `mask`, `drain`,
`mask+drain`. See [Parsers](parsers.md) and [Pre-processing](preproc.md)
for the implementations.

## Sweep everything

```sh
julia --project benchmarks/loghub2/sweep.jl > BASELINES.md
```

The canonical sweep output is committed at
[`benchmarks/loghub2/BASELINES.md`](https://github.com/swiesend/LogClustering.jl/blob/main/benchmarks/loghub2/BASELINES.md).

## Headline numbers (HDFS_2k, 2000 lines)

| parser     | NMI% | ARI% | GA%   | FGA%   | FTA%  | wall (s) |
|------------|-----:|-----:|------:|-------:|------:|---------:|
| identity   | 53.2 |  0.0 |  0.10 |   0.00 |  0.20 |    0.000 |
| num_mask   | 62.0 | 31.7 |  0.10 |  34.71 |  0.35 |    0.003 |
| drain      | 99.9 |100.0 | 99.75 | 100.00 | 83.87 |    0.004 |
| mask+drain | 99.9 |100.0 | 99.75 | 100.00 | 83.87 |   11.272 |

Matches published Drain numbers on HDFS (NMI 99.92, GA 99.75).
`mask+drain` costs ~11 s per 2 k lines because `parse_line` crosses
the FFI boundary once per line; a batched `parse_lines` entry would
amortise the crossing.
