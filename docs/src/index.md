# LogClustering.jl

A modernised port of the 2018 thesis
*Event-Log Analyse mittels Clustering und Mustererkennung*
(Wiesendahl, DLR / TH Köln). The package reconstructs the full thesis
pipeline — framing, tokenisation, template mining, embedding,
clustering, episode mining, sequence forecasting, anomaly scoring —
on modern Julia tooling, with two hot inner loops in Rust (the
recursive regex-cascade parser and the log-key regex inference).

See [Vision](vision.md) for the long-term direction and
[Plan 001](plan-001.md) for the staged decisions.

## Install

```julia
julia> using Pkg
julia> Pkg.add(url = "https://github.com/swiesend/LogClustering.jl")
julia> using Pkg; Pkg.build("LogClustering")   # builds rust/
julia> using LogClustering
```

Cold `using` takes ~1.5 s after precompilation; every hot path (Framing,
Drain, DeepKATE forward, kmeans, metrics, compression) is
first-call-free thanks to `PrecompileTools`.

## Quick smoke test

```julia
using LogClustering
using LogClustering.Framing: parse_frame
using LogClustering.Masking: mask_line

f = parse_frame("<165>1 2024-04-17T10:00:00Z app evntslog 1 ID47 " *
                "[tag@1 k=\"v\"] user=alice login ok")
f.source                           # SOURCE_SYSLOG_5424
mask_line(String(f.message))       # "user=alice login ok"
```

End-to-end demos live under [`examples/`](https://github.com/swiesend/LogClustering.jl/tree/main/examples):

- `template_mining.jl` — typed-slot parsing, skeleton clustering,
  regex inference with class map.
- `anomaly_demo.jl`    — AE training + outlier scoring.
- `sequence_forecast.jl` — SeqLSTM next-event prediction.
- `episode_hueset.jl`  — MT-Span / MV-Span episode mining.

## Layout

- [Pre-processing](preproc.md) — Framing, Masking, Dedup.
- [Parsers](parsers.md)        — Drain3.
- [Models](models.md)          — KATE, DeepKATE, SeqLSTM.
- [Clustering](cluster.md)     — k-means, UMAP+HDBSCAN, sparsity.
- [Mining](mining.md)          — MV-Span, MT-Span.
- [Anomaly](anomaly.md)        — instance-level reconstruction scoring.
- [Evaluation](metrics.md)     — LogHub-2.0 metrics, compression, CV.
- [Rust FFI](rust.md)          — `parse_line`, `infer_regex`.
- [Benchmarks](benchmarks.md)  — LogHub-2.0 baseline sweep.
