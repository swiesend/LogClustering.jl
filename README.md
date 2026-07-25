# LogClustering.jl

Log-event parsing, embedding, clustering, episode mining, and anomaly
detection. Port + modernisation of the 2018 thesis
*Event-Log Analyse mittels Clustering und Mustererkennung* (DLR / TH Köln);
see [`docs/vision.md`](docs/vision.md) and [`docs/plans/001_upgrade-to-2026-stack.md`](docs/plans/001_upgrade-to-2026-stack.md).

## Quick start

```sh
# Julia 1.10 via juliaup; Rust via rustup.
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.build("LogClustering")'
julia --project -e 'using LogClustering'   # ≈1.5 s after precompile
```

See [`examples/`](examples) for end-to-end demos (template mining,
anomaly detection, sequence forecasting, episode mining) and
[`benchmarks/loghub2/`](benchmarks/loghub2) for the LogHub-2.0 bench
harness.

## Automatic differentiation

The package is AD-backend-agnostic — every loss is just plain Julia +
`ChainRulesCore.@ignore_derivatives` on the non-differentiable
boundaries (KATE's sort, DeepKATE's stop-gradient targets). The
**tested** backend is [`Zygote`](https://github.com/FluxML/Zygote.jl);
[`DifferentiationInterface`](https://github.com/JuliaDiff/DifferentiationInterface.jl)
is a dep so users can swap in Enzyme, Mooncake, or ReverseDiff without
changing the loss code.

Regression tests (`test/test_ad.jl`) verify every loss against
finite differences — the numerical gold standard — so a future
backend swap that passes `julia --project -e 'using Test;
include("test/test_ad.jl")'` agrees with the math, not just with
itself.

Enzyme isn't currently wired up because it segfaults during precompile
in the sandbox this branch was developed in; the code is structured
so adding it is a one-line change:

```julia
using DifferentiationInterface: AutoEnzyme, gradient
g = gradient(loss, AutoEnzyme(), ps)
```

## Embedders (Stage C)

Three Lux autoencoders / training recipes ship:

| module        | what it is | cluster-ID source |
|---------------|------------|--------------------|
| `DeepKATE`    | thesis §3.2.3, K-competitive + sine bottleneck | `Sparsity.sparsity_clusters` (top-k active neurons) |
| `VQVAE`       | van den Oord 2017, discrete codebook bottleneck | `VQVAE.assign_codes` (argmin distance) |
| `SimCSE`      | Gao 2021 InfoNCE — a training *recipe* over any Lux encoder | whatever clusterer the caller picks |

`SimCSE.simcse_step_loss` runs the encoder twice so Dropout's RNG
yields a `(h_i, h_i⁺)` positive pair for contrastive learning. `VQVAE.
assign_codes` is the plan's "cluster id for free".

## CLI

```sh
bin/logcluster --help                 # list subcommands
bin/logcluster mask --data log.txt    # typed-slot masking
bin/logcluster train --kind drain      --data log.txt --out model.jld2
bin/logcluster train --kind deep_kate  --data log.txt --out ae.jld2    --auto
bin/logcluster train --kind vq_vae     --data log.txt --out vq.jld2    --auto
bin/logcluster train --kind seq_lstm   --data log.txt --out lstm.jld2  --auto
bin/logcluster classify --model model.jld2 --data newlog.txt --format tsv
bin/logcluster score    --detector det.jld2 --data newlog.txt
bin/logcluster benchmark data/HDFS_2k.log_structured.csv drain
bin/logcluster download-loghub --all

# Production streaming + rule-based triggers (see docs/deployment.md).
bin/logcluster stream --rules examples/rules-production.json --tail /var/log/syslog
bin/logcluster rules  --print-defaults > /etc/logcluster/rules.json
```

For long-running deployments see [`docs/deployment.md`](docs/deployment.md)
(Docker + systemd) and [`docs/json-schema.md`](docs/json-schema.md)
(stream wire format).

Every subcommand also works in-process: `LogClustering.CLI.main(["mask",
"--data", path])` returns an exit code without shelling out.

## Is Drain part of the pipeline?

No — Drain ships as a **comparison baseline**, not a component of the
thesis pipeline. The canonical thesis pipeline is
`Framing → Masking → parse_line → cluster → infer_regex`, with
`DeepKATE` / `SeqLSTM` / `Instance` on the embedding side. Drain is
wired into `benchmarks/loghub2/run.jl` so the LogHub-2.0 sweep can
quantify the gap between our stack and the field's deterministic
standard. The sweep now covers **all 16 LogHub-2.0 2k subsets** —
see [`BASELINES.md`](benchmarks/loghub2/BASELINES.md) for the
full table (seven parsers × sixteen datasets = 112 rows).

The `our_stack` parser combines both — Drain for grouping, our
`infer_regex` (anti-unified) for template rendering — and matches
Drain on every *clustering* metric (GA 99.75, NMI 99.92 on HDFS).
The batched `parse_lines` FFI makes every `mask`-prefixed path run
in well under `0.03 s` per 2 k lines (400–1000× faster than the
per-line FFI crossings it replaced).

## Root-cause-analysis in one command

Once a model is trained (or found in the registry by
`logcluster select-model`), `logcluster rca` composes
clustering → anomaly scoring → episode mining into a single
report:

```sh
logcluster train --kind deep_kate --data corpus.log \
    --auto --reuse --registry ~/.cache/logclustering/models \
    --out /tmp/model.jld2

logcluster rca --model /tmp/model.jld2 --data corpus.log \
    --percentile 0.05 --topk 10 --format md --out /tmp/rca.md
```

The top-N ranked root-cause episodes — scored by
`support · mean_anomaly_density` — are the sequences of cluster
ids that fire often *and* fire in anomalous windows. The
Markdown output includes representative lines per pattern so the
runbook entry writes itself. See
[`docs/src/rca.md`](docs/src/rca.md) for flag reference and
[`src/RCA.jl`](src/RCA.jl) for the Julia API.

## Model reuse across investigations

Saved DeepKATE / VQ-VAE / SeqLSTM bundles carry a
`corpus_fingerprint` — a SHA-256 of the sorted vocabulary. The
`--reuse --registry DIR` flag on `logcluster train` scans that
directory for a bundle whose fingerprint matches the new corpus
and, on a hit, copies it to `--out` without running SGD. The
same mechanism drives `logcluster select-model`, which prints
the matched path (or exits non-zero) so it's scriptable.

## Value-level outlier detection

Typed-slot masking replaces every `<IP>` / `<INT>` / `<TIMESTAMP>` before
clustering, so a never-before-seen IP or a wildly out-of-range
number is invisible to the template-reconstruction score. To catch
those, use [`Masking.mask_lines_with_values`](src/PreProc/Masking.jl)
— it returns *both* the templated text (for the AE) and the captured
slot values (for a separate value-novelty head):

```julia
using LogClustering.Masking: mask_lines_with_values
using LogClustering.Instance: ValueNoveltyDetector, update!, combined_anomaly

templates, values = mask_lines_with_values(lines)
# Templates go into feature extraction → `X = …`
# Values feed a per-label seen-set + running-mean/variance detector:
det = ValueNoveltyDetector()
for vs in train_values; update!(det, vs); end

# Fuse template-reconstruction and value-novelty into one score per line.
scores = combined_anomaly(model, ps, st, X, test_values, det;
                          weights = (template = 0.5, values = 0.5))
```

`weights = (template = 0, values = 1)` gives pure value-outlier
scoring; `(1, 0)` recovers plain reconstruction. Numeric z-score
is computed against each label's empirical mean/variance; categorical
values are flagged when unseen.

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
open docs/build/index.html
```

The `Documentation` CI job deploys the same output to GitHub Pages on
every push to `main`.

## Dev-loop tips

- **Full suite via `Pkg.test()` is slow** (~2 min) because it rebuilds
  a sandbox env every call. For iteration use:

  ```sh
  julia --project -e 'using Test; include("test/test_framing.jl")'   # one file, ≈2 s
  julia --project -e 'using Test; include("test/runtests.jl")'       # all files, ≈1:20
  ```

  `LogClustering` is `PrecompileTools`-instrumented: first `using`
  after a source change takes ~20 s, then subsequent loads are ≈1.5 s
  and every hot path (parse_frame, DeepKATE forward, SeqLSTM,
  kmeans_cluster, metrics, compression) is first-call-free.

- **Python side** (UMAP, HDBSCAN, embedders) is optional. `cd py &&
  uv sync` once, then
  ```sh
  export JULIA_CONDAPKG_BACKEND=Null
  export JULIA_PYTHONCALL_EXE="$PWD/py/.venv/bin/python"
  ```
  before `julia --project`. Without these, `using LogClustering`
  still works; Python-only calls (`umap_reduce`, `hdbscan_cluster`)
  error with remediation steps when invoked.

- **Rust side** is built by `deps/build.jl` via `cargo build
  --release`. If `cargo` isn't on `PATH`, the Rust module gracefully
  degrades — everything else keeps working. `cd rust && cargo test`
  runs the crate's own unit tests in ≈0.3 s.

## Layout

```
src/
  KATE.jl, DeepKATE.jl          # Lux ports of the thesis autoencoders
  PreProc/Framing.jl            # recursive-descent envelope parser
  Mining/Episodes.jl            # MV-Span / MT-Span (thesis §3.2.7)
  Models/SeqLSTM.jl             # LSTM + peephole + bidirectional
  Anomaly/Instance.jl           # reconstruction-error scoring (§3.2.5)
  Cluster/{Pipeline, Sparsity}.jl
  Eval/{Metrics, Compression, Harness, CV}.jl
  Rust.jl                       # @ccall bindings for rust/logclustering_rs
rust/
  src/{lib, infer, parse_line}.rs   # parse_line + log-key regex inference
py/                              # uv-managed PythonCall companion
benchmarks/loghub2/              # Stage F benchmark harness + download.jl
examples/                        # end-to-end demos
docs/                            # vision + plan
```
