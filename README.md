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
