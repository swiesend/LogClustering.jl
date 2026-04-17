# Plan 002 — Persistence, AutoTune, CLI

| Field | Value |
|---|---|
| Status | Proposed |
| Scope | Make every trained artifact in the repo reproducible, right-sized from data, and usable from a shell without Julia-on-the-hot-path |
| Supersedes | — |
| See also | [`001_upgrade-to-2026-stack.md`](001_upgrade-to-2026-stack.md) |

## Context

Plan 001 built the algorithmic surface — framing, masking, Drain,
DeepKATE, VQ-VAE, SimCSE, SeqLSTM, episode miners, anomaly heads,
the LogHub-2.0 harness. None of it is usable *in practice* yet:

- Trained models live for the lifetime of one Julia session. There's
  no way to ship a fitted `DeepKATE` / `VQVAE` / `SimCSE` / `Drain3`
  / `ValueNoveltyDetector` to a consumer.
- Every hyperparameter is picked by the caller from the defaults.
  `deep_kate(n)` wants a `latent`, a `k1`, a `hidden`; users have
  to guess. The thesis's "intrinsic dimensionality → latent size"
  intuition isn't captured in code.
- Everything ships as library API. A practitioner who just wants to
  point a tool at a log file and get templates + cluster ids + an
  anomaly score has to write Julia.

This plan closes the gap with three stackable modules.

## Decision log

### Stage A — `src/Persistence.jl`  *(must land first)*

One save/load API across every persistable artifact in the repo.
Backing format: **JLD2** (Julia-native, Apache-2.0, widely used in
SciML, handles custom types + `NamedTuple`s cleanly).

**Bundle layout.** Every saved artifact is a `PersistedBundle`:

```julia
struct PersistedBundle
    kind::Symbol           # :deep_kate, :vq_vae, :simcse_encoder,
                           # :seq_lstm, :kate, :drain, :value_novelty,
                           # :dedup, :chain (generic Lux Chain)
    lc_version::VersionNumber   # the LogClustering.jl version at save time
    schema_version::UInt8       # the bundle schema; bumped on ABI change
    spec::NamedTuple            # constructor kwargs: recreates the
                                # model structure via the kind-specific
                                # rehydrator
    payload::Any                # kind-dependent: (ps, st) for Lux,
                                # the whole struct for Drain etc.
    metadata::Dict{String, Any} # freeform — training dataset, date,
                                # git SHA, sweep row, user notes
end
```

**Public API** (`src/Persistence.jl`):

```julia
save(path, artifact; metadata = Dict{String,Any}()) -> Nothing
load(path) -> PersistedBundle
rehydrate(bundle::PersistedBundle) -> artifact   # inverse of save
```

- `save(path, chain::Lux.Chain, ps, st; spec, kind, metadata)` — the
  general Lux path. `spec` captures the constructor kwargs so
  `rehydrate` can call the right factory to rebuild the graph.
- `save(path, d::Drain3.Drain; metadata)` — serialises the whole
  mutable struct (`depth`, `sim_th`, `clusters`, `root` tree). The
  `parametrize` closure doesn't round-trip; we store a symbol
  (`:digit`, `:numeric`, `:none`, `:custom`) and a callable-registry
  lookup table on load. Unknown symbols error with a clear message.
- `save(path, det::ValueNoveltyDetector; metadata)` — ditto,
  straightforward fields.
- `save(path, s::DedupState; metadata)` — straightforward.

**Rehydrators.** One function per `kind`, registered in a dispatch
table that lives in `Persistence` and is extended by each model
module via `Persistence.register_rehydrator!(:deep_kate, fn)` on
module load.

**Schema versioning.** `schema_version` starts at `1`; any breaking
bundle-layout change bumps it. `load` on a newer schema than the
current build errors out with an upgrade note; an older schema goes
through a migration function.

**Files (new).**
- `src/Persistence.jl` — API + JLD2 glue + kind dispatch.
- Each existing module gains a short `_register_persistence()`
  helper called from its own `__init__`.
- `test/test_persistence.jl` — round-trip every model kind.

**Dependencies.** Add `JLD2` to `Project.toml`. No runtime cost: JLD2
only loads when `save` / `load` is first called (weak-link via
`Requires.jl`-style lazy import, or just a hard dep — JLD2 is ~1 s
to precompile).

**Verification.**
1. `using Test; include("test/test_persistence.jl")` — every kind
   round-trips (construct → train one step → save → load → forward
   must match bitwise).
2. Save a DeepKATE trained on HDFS_2k; load in a fresh Julia session;
   reproduce the same `assign_codes` / `anomaly_score` vector to
   within float tolerance.
3. `save` / `load` complete in < 2 s for the default models.

---

### Stage B — `src/AutoTune.jl`

Heuristic-first hyperparameter picker. Expensive search is opt-in.

**Public API.**

```julia
AutoTune.fit_hyperparams(kind::Symbol, corpus; budget = 0, kwargs...) ->
    NamedTuple  # drop-in kwargs for the kind's factory
```

`corpus` is whatever the model wants at training time — a `Matrix` for
DeepKATE/VQVAE/KATE, a `Vector{Matrix}` of `(T, batch)` id windows
for SeqLSTM/SimCSE.

**Default heuristics.** Cheap + data-driven:

| kind         | latent / embed | hidden                | other                    |
|--------------|----------------|-----------------------|--------------------------|
| `:deep_kate` | PCA elbow (95 % var) capped `[2, 16]` | `2 · vocab / 3` capped `[32, 512]` | `k1 = round(0.25 · hidden)`, even |
| `:vq_vae`    | PCA elbow, capped `[4, 64]`    | `max(64, 2·embed)`    | `codebook_size = clamp(⌈√n⌉, 16, 256)` |
| `:seq_lstm`  | `embed = clamp(log2(vocab)·4, 8, 64)` | `embed · 4`          | —                        |
| `:simcse`    | same as DeepKATE latent                | same as DeepKATE hidden | τ = 0.05 |

**PCA elbow helper.** `AutoTune.pca_elbow(X; var_threshold = 0.95)`
returns the smallest `k` such that cumulative singular-value variance
reaches the threshold. Uses `LinearAlgebra.svd` — no new dep.

**Optional search (`budget > 0`).** Random search over the unset
fields against a held-out split from `Eval.CV.time_ordered_split`,
minimising reconstruction loss (for AEs) or InfoNCE (for SimCSE) or
NLL (for SeqLSTM). Budget is an integer number of trials; the
defaults seed the first trial. No Bayesian optimisation yet — random
is sufficient up to ~50 trials and depends on zero extra packages.

**Files (new).** `src/AutoTune.jl`, `test/test_autotune.jl`.

**Verification.**
1. Heuristics return numbers in documented ranges on random inputs.
2. On HDFS_2k tokenised via `mask_lines` + BoW, `fit_hyperparams(:deep_kate, …)`
   picks `latent` ≤ 8 (HDFS's intrinsic structure has ~6 dimensions).
3. `budget = 10` random search on a tiny synthetic dataset drops
   training loss vs the default-only picks — verifies the loop,
   not a particular improvement ratio.

---

### Stage C — `bin/logcluster` (CLI)

A single-binary-flavoured entrypoint. Uses **Comonicon.jl** —
stdlib-flavoured arg parsing + auto-generated `--help` + optional
PackageCompiler integration for an actual binary later.

**Top-level usage.**

```
logcluster [--project PATH] <command> [args...]
```

**Subcommands.**

| command            | purpose |
|--------------------|---------|
| `train`            | Fit any model. `--kind deep_kate \| vq_vae \| simcse \| seq_lstm \| drain`, `--data FILE`, `--epochs N`, `--out model.jld2`, `--auto` (use `AutoTune.fit_hyperparams`). |
| `classify`         | Load a saved model, emit `(line, cluster_id, template)` per input line. `--model model.jld2 --data FILE --format tsv\|json`. |
| `score`            | Anomaly score per line. `--model model.jld2 --detector det.jld2 --data FILE`. |
| `benchmark`        | Run one parser against one LogHub `*_structured.csv` (thin wrapper over existing `benchmarks/loghub2/run.jl`). |
| `download-loghub`  | Fetch the 2k subsets (thin wrapper over `benchmarks/loghub2/download.jl`). |
| `mask`             | Apply the typed-slot battery to stdin or `--data`, emit masked lines + optional `--values` JSON stream. |

**Streaming.** All subcommands read lines from a file or stdin; write
to `--out` or stdout. Input frames default to raw; `--framed` applies
`Framing.parse_frame` first so syslog / CRI / Docker are all handled.

**Persistence hooks.** `train` writes a `PersistedBundle` via
`Persistence.save`. `classify` / `score` load with `Persistence.load` +
`rehydrate`.

**AutoTune integration.** `train --auto` dispatches through
`AutoTune.fit_hyperparams` on a held-out slice; `--auto --budget 20`
triggers random search.

**Files (new).**
- `bin/logcluster` — shell shim (`#!/usr/bin/env -S julia --project`)
  that calls into `CLI.main`.
- `src/CLI.jl` — Comonicon entry with one `@cast` per subcommand.
- `test/test_cli.jl` — invoke each subcommand in-process against a
  small fixture (no shelling out), assert exit codes and output
  shape.

**Dependencies.** Add `Comonicon` and `JSON3` to `Project.toml`.

**Binary (optional).** Comonicon ships `deps/build.jl` glue for
`PackageCompiler.jl` when `LogClustering.build_binary()` is called
— a follow-up commit, not required for 002.

**Verification.**
1. `bin/logcluster --help` lists all subcommands.
2. `bin/logcluster train --auto --kind deep_kate --data <HDFS_2k> --out /tmp/m.jld2`
   and then `classify --model /tmp/m.jld2 --data <HDFS_2k> --out /tmp/out.tsv`
   together produce a deterministic (model, templates) pair.
3. End-to-end walltime on a 2 k-line fixture < 5 s for `mask`,
   < 20 s for `train --auto deep_kate`.

---

## Global verification gates

1. Every model added to Plan 001 Stage C round-trips via
   `Persistence.save` / `load`.
2. `AutoTune.fit_hyperparams(kind, X)` returns a `NamedTuple` that
   the `kind`'s factory accepts as kwargs without further edits.
3. `bin/logcluster train` followed by `classify` (or `score`) on a
   fresh process reproduces the within-process result.
4. All existing Plan-001 tests still pass (no regressions in
   `runtests.jl`).
5. CI gains a `cli` job that builds the package and runs one
   end-to-end round-trip on the toy fixture.

## Module map after this plan

Additions only (everything from Plan 001 survives untouched):

```
src/
  Persistence.jl              # save/load + rehydrators
  AutoTune.jl                 # heuristics + optional random search
  CLI.jl                      # Comonicon entry + subcommand dispatch
bin/
  logcluster                  # shell shim -> CLI.main
test/
  test_persistence.jl
  test_autotune.jl
  test_cli.jl
```

## Sequencing

1. **Persistence first.** Every other feature depends on it. Ship as
   one commit: API + registrar + round-trip tests across all model
   kinds.
2. **AutoTune second.** Small heuristic-only commit; add the
   random-search branch in a follow-up once the heuristics prove
   themselves.
3. **CLI last.** Depends on both. Ship with every subcommand
   end-to-end tested on the toy fixture; binary compilation via
   PackageCompiler is a separate follow-up PR.

## What is *not* part of this plan

- A model registry / central model-zoo (superset of Persistence;
  out of scope — per-artifact save/load is what practitioners need).
- Cloud / remote model storage (S3, HTTP).
- Online fine-tuning of a loaded model — `train` always trains from
  scratch; incremental training across save/load sessions is a
  follow-on that needs AD-state round-tripping.
- A GUI or web frontend.
- Streaming ingestion (Kafka / Vector / OTEL) — still handled by
  Plan 001's "Out of scope" list.
