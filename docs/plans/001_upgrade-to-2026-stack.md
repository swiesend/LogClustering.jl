# Plan 001 — Upgrade to the 2026 stack

| Field | Value |
|---|---|
| Status | In progress (Stage A complete; thesis ports landing into B/C/E″) |
| Scope | Revive `LogClustering.jl` and realise the full thesis pipeline on modern tooling |
| Supersedes | — |
| See also | [`../vision.md`](../vision.md), the 2018 thesis ([DLR elib 126129](https://elib.dlr.de/126129/)) |

## Language boundary

Main implementation is Julia. Performance-critical kernels — the recursive
regex-cascade parser (thesis Algorithm 3.1) and the log-key regex
inference (Algorithm 3.10) — are written in Rust in the companion crate
`rust/logclustering_rs`, exposed through `@ccall` in `src/Rust.jl`. The
crate uses `regex`/`regex-automata` (RE2-flavoured, linear-time matching)
so the kernels stay composable with downstream Hyperscan/RE2 emit.
`deps/build.jl` invokes `cargo build --release`; if `cargo` is absent the
Rust module degrades to a runtime error and the rest of the package
still loads. CI builds the crate in a dedicated `rust` job
(`cargo fmt --check`, `clippy -D warnings`, `cargo test --release`).

## Context

The vision ([`../vision.md`](../vision.md)) describes what the project wants to
be. This plan records the decisions for *how* we get there, staged so each
stage is independently mergeable and verifiable.

## Decision log

Each stage is a decision; later stages may depend on earlier ones but do not
require them to be fully complete.

---

### Stage A — Revival foundation  *(Started)*

**Decision.** Migrate to Julia 1.10 LTS and **Lux** (not Flux). Replace
`REQUIRE` with `Project.toml`. Preserve KATE's math; modernise only the API.

**Rationale.** Lux's explicit-parameter model (`initialparameters`,
`initialstates`, `layer(x, ps, st)`) is cleaner than Flux's implicit state,
composes better with `Zygote.@ignore_derivatives`, and is the direction the
SciML ecosystem is moving. The k-winner-take-all sort needs a clean
non-differentiable boundary — Lux makes that explicit.

**Concrete steps.**
- [x] Remove `REQUIRE`; add `Project.toml` targeting Julia 1.10 (commit `e94ed31`).
- [x] Rewrite `src/KATE.jl` as an `AbstractLuxLayer` with
  `initialparameters`/`initialstates`; extract `_apply_kcomp` with
  `@ignore_derivatives` around the sort; fix the latent bug where the
  competition was computed but discarded (commit `e94ed31`).
- [x] Replace `test/runtests.jl` `@test 1==2` stub with a real suite:
  construction, parameter shapes, vector + batch forward, unit tests of the
  competition math, Zygote gradient flow, text-preprocessing utilities
  (commit `e94ed31`).
- [x] Convert CI from Travis/AppVeyor to GitHub Actions
  (`.github/workflows/CI.yml`; `appveyor.yml` removed).
- [x] Add `Manifest.toml` after first successful `Pkg.instantiate()` on
  Julia 1.10.11 (corrects stale `LuxCore` UUID).
- [x] Wire a `py/` companion project (uv-managed `pyproject.toml` +
  `uv.lock`, driven through `PythonCall.jl` with
  `JULIA_CONDAPKG_BACKEND=Null` + `JULIA_PYTHONCALL_EXE` pointing at
  `py/.venv`) for embedders, UMAP-learn, HDBSCAN, LILAC.

**Files touched.** `Project.toml`, `src/KATE.jl`, `src/LogClustering.jl`,
`test/runtests.jl`, `test/test_KATE.jl`, `.github/workflows/CI.yml` (new),
`py/` (new).

**Verification.** `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`
passes on Julia 1.10.

---

### Stage B — Deterministic track (baselines + preprocessing)

**Decision.** Port Drain3 as the primary baseline and as a preprocessing step.
Wrap Brain and LogPPT for comparison. Hybrid: Drain templates feed the neural
embedders; sparsity prior operates on slot-variable sequences, not raw tokens.

**Rationale.** Drain (He et al. 2017) is the de-facto standard in 2024-2026
log-parsing benchmarks. A neural model without a deterministic baseline is
unfalsifiable.

**Files (new).** `src/Parsers/{Drain.jl, Brain.jl, LogPPT.jl}`.

**Verification.** Drain round-trip on a fixture; PA/GA match published numbers
on HDFS within 1 pt.

---

### Stage C — Compression / encoder-decoder track *(preserves thesis spirit)*

**Decision.** Keep KATE-modernised as the reference. DeepKATE (the
thesis's own contribution, §3.2.3, Quellcode 3.7/3.8/3.9) is *ported*
from Flux to Lux in `src/DeepKATE.jl`: ten-layer autoencoder, two
`KCompetetive` layers with `k < out` (this required extending
`KCompetetive` to separate `out_dims` from `k`), sine-activated
bottleneck, dropout, and the triplet Pareto loss (prev/ce/succ) with
stop-gradient targets via `@ignore_derivatives`. Add three siblings:

- **VQ-VAE** (van den Oord et al. 2017) — the principled descendant of
  k-winner-take-all; codebook *is* the cluster vocabulary; discrete code
  becomes the cluster ID for free.
- **Masked / denoising autoencoder** (Vincent 2008; He et al. MAE 2022) —
  stronger self-supervised signal than pure reconstruction.
- **Contrastive sentence encoder** (SimCSE, Gao et al. 2021) with log-specific
  augmentations (parameter masking, timestamp dropout).

Replace the regex BoW input with **BPE** via `BytePairEncoding.jl` (or
SentencePiece) trained per-dataset.

**Rationale.** VQ-VAE is the clean 2020s formalisation of what KATE was reaching
for. SimCSE empirically beats reconstruction-trained encoders on clustering.

**Files (new).** `src/Models/{VQVAE.jl, DenoisingAE.jl, SimCSE.jl,
Embedders.jl}`, `src/Data/Tokenizers.jl`.

**Verification.** VQ-VAE codebook perplexity stable; SimCSE NMI on HDFS
≥ KATE-modernised + 3 pts.

---

### Stage D — LLM track

**Decision.** Ship four LLM touch-points, each bounded by a deterministic
verifier:

1. **Embedders** — BGE-m3, GTE, multilingual-E5, Nomic-embed-v1.5,
   Jina-embeddings-v3 — exposed behind one `Embedder` trait, selected by
   config.
2. **LILAC-style parser** — in-context template extraction with retrieved
   demonstrations; small-model fallback: LogPPT.
3. **Cluster labelling** — LLM consumes cluster medoids, returns a
   human-readable name + canonical template.
4. **Closed loop** — LLM proposes template → Drain validates structural
   consistency → residuals re-embedded → repeat until coverage plateau.

**Rationale.** LLMs without a deterministic verifier hallucinate templates.
Keeping Drain / the regex verifier authoritative gives us the LLM's
generalisation without its drift.

**Files (new).** `src/Parsers/LILAC.jl`, `src/LLM/{Loop.jl, Labeling.jl}`,
`py/{embedders.py, lilac_parser.py}`.

**Verification.** Closed-loop residual-coverage curve plateaus within ≤ 5
iterations on HDFS (regression test).

---

### Stage E — Episode mining  *(thesis miners landed)*

**Decision.** `src/Mining/Episodes.jl` ships the thesis's two serial-episode
miners (ported from Quellcode 3.11–3.14):

- **MV-Span** — SPADE-style depth-first prefix-growth over a
  pseudo-projected vertical database. Constraints: `min_sup`,
  `min_utility` (external / local / average), `max_repetitions`,
  `max_gap`, `max_time_duration`; `result_set ∈ {:all, :closed}`.
- **MT-Span** — TSpan-derived prefix-growth with EWU + IESC upper-bound
  pruning. Returns `(moSet, hueSet)` — minimal occurrences and the
  high-utility episode set.

Both accept seed `prefixes`, and both are pure-Julia (no Rust crossing)
since the inner loops are already vector-index operations on `Vector{Int}`.
PrefixSpan / SPADE / CM-SPADE remain on the roadmap as external
baselines to compare against.

### Stage E — Clustering & downstream

**Decision.** Standard pipeline for every embedder: L2-normalise → UMAP(n=15,
min_dist=0.0) → HDBSCAN(min_cluster_size=10). Compare k-means and
"clustering-via-sparsity" (KATE hidden-layer argmax) to isolate the thesis's
original contribution.

**Rationale.** McInnes/Healy UMAP + HDBSCAN is the 2024-2026 default;
sparsity-as-clustering lets us ask whether KATE's bottleneck *is* its own
clusterer.

**Files (new).** `src/Anomaly/Sequence.jl`.

**Clustering pipeline ported.** `src/Cluster/Sparsity.jl` realises the
thesis's native "clustering-via-sparsity" from KATE's competitive
bottleneck — `sparsity_clusters(Z, k)` reads off each sample's top-k
active-neuron indices as a discrete cluster label (optional `signed=true`
keeps positive and negative winners distinct). `src/Cluster/Pipeline.jl`
provides `l2_normalise` + `kmeans_cluster` (via `Clustering.jl`) as the
first distance-based comparator; `hdbscan_cluster` is a lazy
`PythonCall → hdbscan` shim that raises a remediation-step error when
the uv venv under `py/` isn't configured. UMAP plugs in the same way.

**Instance anomaly ported.** `src/Anomaly/Instance.jl` implements the
thesis's §3.2.5 reconstruction-error scoring on top of any Lux
autoencoder (matched to DeepKATE by default): per-sample `E_abs`
(Gleichung 3.3), squared variant, latent-space distance to an optional
reference, and the weighted "Meta-Metrik" `anomaly_score`. All forward
passes run in testmode so dropout and KATE's competitive redistribution
are disabled at inference.

**Verification.** Three clustering methods produce comparable NMI on HDFS
(within 5 pts).

---

### Stage E′ — Pre- and post-processing pipeline

**Decision.** Replace `transform_text_to_input` (a single regex split plus
log-count normalisation) with a layered, reorderable pipeline.

**Pre-processing.**
1. **Structural framing** — strip collector prefix (syslog RFC-5424,
   journald, Kubernetes CRI, Docker JSON-lines). Grammar registry in
   `PreProc/Framing.jl`. *Initial pass landed*: zero-copy
   recursive-descent parser for RFC 5424 (incl. nested STRUCTURED-DATA
   with escaped PARAM-VALUEs), Kubernetes CRI, and Docker json-file;
   `parse_frame` dispatches on the first byte and returns a `Frame` of
   `SubString` views. Hot-path allocation is constant in input length
   (≤ 512 B per call, dominated by the returned `Frame` itself).
   Journald export format is still TODO.
   **Thesis-body parser also ported** — `src/Rust.jl` exposes
   `parse_line`, the recursive regex-cascade descent of thesis
   Algorithm 3.1 (ordered label battery, left/right recursion on
   unmatched slices). Lives in the Rust crate because it's the hot
   inner loop; Julia side returns `Vector{Span}` with 1-based byte
   indices and the matched label (or `nothing` for raw runs).
2. **Unicode NFKC** + optional case folding (preserve case for identifiers).
3. **Typed-slot masking** — ordered battery of pre-compiled regexes (IP/IPv6,
   MAC, UUID, SHA-1/256 hex, hex-address, paths, URLs, email, ISO/Unix
   timestamps, ANSI-colour codes, numeric literals with units, quoted
   strings). Order matters (timestamp before number; UUID before hex). Each
   mask emits a typed placeholder `<IP>`, `<NUM>`, `<PATH>` — downstream
   metrics and embedders benefit from the type signal.
4. **Tokenisation** — BPE/SentencePiece learned on the masked corpus; typed
   placeholders become single tokens.
5. **Content-hash deduplication** — streaming xxHash64 + Bloom filter. 90 %+
   of production log volume is exact duplicates; dedup before clustering.

**Post-processing.**
1. **Slot-purity re-check** — if a slot's realisations across the cluster
   have entropy below τ (say 0.2 bits), promote back to literal.
2. **Slot typing** — run the typed-regex battery over each slot's
   realisations; assign the most-specific type that matches ≥ 95 %.
3. **Template canonicalisation** — sort alternatives lexicographically,
   collapse whitespace, consistent placeholder style. Hash the canonical form
   for cross-parser deduplication.
4. **Cluster merge** — post-hoc merge clusters whose canonical templates have
   edit-distance ≤ 1 token and whose embedding centroids cosine-match ≥ 0.98.
5. **Regex compilation cache** — hash-indexed compiled `Regex`/Hyperscan DB,
   shared across evaluation runs.

**Rationale.** Log-specific tokens (IPs, paths, hex) must survive tokenisation
intact; downstream embedders cannot undo a botched split. Dedup is free
throughput. Post-processing fixes Drain's over-splitting without retraining.

**Files (new).** `src/PreProc/{Framing.jl, Masking.jl, Dedup.jl}`,
`src/PostProc/{Purity.jl, Typing.jl, Canonical.jl, Merge.jl}`.

**Verification.** Masking is idempotent (`mask ∘ mask = mask`);
canonicalisation is a retraction (`canon ∘ canon = canon`); dedup Bloom
false-positive rate ≤ 1e-4 on a 1 M-line probe.

---

### Stage E″ — Performant generic-regex generation

**Thesis baseline already ported.** `src/Rust.jl` exposes `infer_regex`,
a direct port of thesis Algorithm 3.10: group tokens by position, collapse
agreements to literals, disagreements to `(a|b|c)` alternations, rewrite
descriptive placeholders (`%RCE_DATETIME%` etc.) to a configurable
wildcard. Metacharacter escaping is via the `regex` crate's `escape`. A
parity test against the thesis's own Beispiel 3.1 output is in
`test/test_rust.jl`. The full Stage E″ pipeline below *layers on top of*
the thesis baseline; anti-unification replaces the position-aligned
grouping when lines differ in length.

**Decision.** Turn a cluster of raw lines or a template with `<*>` slots into
a single, precise, performant regex via:

1. **Anti-unification over aligned tokens** — token-level greedy LCS across
   deduplicated cluster representatives (m ≤ ~50). Positions where all lines
   agree → literal; positions where they disagree → slot. Deterministic.
2. **Per-slot regex inference (MDL ladder)** — for each slot, pick the
   tightest regex from an increasingly general ladder: exact-literal → enum
   `(a|b|c)` (if |S| ≤ K) → typed class (IP/UUID/NUM/HEX/PATH) → bounded
   `[A-Za-z0-9_.-]{min,max}` → unbounded.
3. **Alternation minimisation** — merge sibling templates that differ in
   exactly one slot; run to fixpoint.
4. **Emit RE2** by default — linear-time, no backrefs, no lookaround. No
   catastrophic backtracking by construction.
5. **Multi-pattern hot path** — compile the full regex set into a single
   **Hyperscan** database (Intel, SIMD, multi-GB/s). Fallback: `Automa.jl`
   (pure-Julia DFA source-gen) or Go `regexp`.
6. **Verification pass** — each regex must match 100 % of its training
   cluster and ≤ ε of negatives from other clusters. Failures tighten the
   slot class and retry.
7. **Cost monitoring** — reject candidates whose compiled NFA state count or
   per-match cost exceeds a budget; fall back to a simpler template.

**LLM assist (bounded).** After deterministic generation, ask an LLM for a
tighter slot class; accept only if the candidate matches 100 % of the cluster
and stays RE2-compatible. LLM proposes, deterministic verifier decides.

**Rationale.** RE2/Hyperscan give worst-case linear matching. Hyperscan scans
the entire template set in one pass — throughput scales with *characters*, not
templates. MoLFI-style evolutionary search is explicitly rejected: slow and
unnecessary once anti-unification + typed slots are in place.

**Files (new).** `src/Regex/{AntiUnify.jl, SlotLadder.jl, Minimize.jl,
Emit.jl, Hyperscan.jl, Verify.jl}`.

**Verification.**
- Correctness: generated regex matches 100 % of training cluster and ≤ ε on
  10 k held-out negative lines.
- Performance: generation < 1 ms per 100-line cluster; Hyperscan single-thread
  throughput ≥ 500 MB/s against ≥ 10 k compiled templates.
- Recorded in `benchmarks/regex/`.

---

### Stage F — Evaluation  *(harness landed)*

**Decision.** LogHub-2.0 (Zhu et al. ISSRE 2023) as the single source of
ground truth — all 14 datasets, ~50 k annotated lines each.

**Metrics ported.** `src/Eval/Metrics.jl` implements the four parsing
metrics (`parsing_accuracy`, `group_accuracy`, `grouping_f1` ↔ FGA,
`template_group_f1` ↔ FTA) and the four clustering metrics
(`normalised_mutual_information`, `adjusted_rand_index`, `purity`,
`v_measure`). Pure Julia, dependency-free, and accept both integer
label vectors and raw template strings.

**Harness landed.** `src/Eval/Harness.jl` + `benchmarks/loghub2/run.jl`
load a LogHub-2.0 `*_structured.csv` (dependency-free, RFC-4180–aware
CSV reader), run a parser function `Vector{String} → Vector{String}`,
and print a one-row TSV of all ten metrics. Three sanity baselines
ship out of the box (`identity`, `constant`, `num_mask`); adding a
parser is one entry in the `PARSERS` dict. Compression metrics
(BPC, codebook perplexity, dictionary size) and `Eval/CV.jl` remain
to land.

**Baselines table** (in `README.md`): Drain, Brain, LogPPT, LILAC,
BGE+HDBSCAN, KATE-original, KATE-modernised, VQ-VAE, SimCSE-logs across all
14 datasets.

**Reproduction gate.** KATE-modernised matches the original KATE paper's
20NG clustering NMI within ±2 pts as a sanity check.

---

## Module map after this plan

```
src/
  KATE.jl                      # modernised (Stage A, done; k ≠ out)
  DeepKATE.jl                  # ported from Flux to Lux (thesis §3.2.3)
  Rust.jl                      # @ccall bindings for rust/logclustering_rs
  LogClustering.jl             # umbrella
  Data/{LogHub.jl, Tokenizers.jl, EventLog.jl, Corpus.jl}
  PreProc/{Framing.jl, Masking.jl, Dedup.jl}
  PostProc/{Purity.jl, Typing.jl, Canonical.jl, Merge.jl}
  Regex/{AntiUnify.jl, SlotLadder.jl, Minimize.jl, Emit.jl,
         Hyperscan.jl, Verify.jl}
  Parsers/{Drain.jl, Brain.jl, LogPPT.jl, LILAC.jl}
  Models/{VQVAE.jl, DenoisingAE.jl, SimCSE.jl, Embedders.jl,
          SeqLSTM.jl (ported, thesis §3.2.8), SeqTransformer.jl, SeqMamba.jl}
  Cluster/{Pipeline.jl, Sparsity.jl}
  Mining/{Episodes.jl}              # MV-Span / MT-Span (thesis §3.2.7, ported)
  Anomaly/{Instance.jl (ported, thesis §3.2.5), Sequence.jl}
  RCA/{Graph.jl, CausalDiscovery.jl, LLM.jl}
  Eval/{Metrics.jl, Harness.jl, CV.jl}
  LLM/{Loop.jl, Labeling.jl}
rust/                          # cdylib: parse_line + infer_regex (thesis §3.2.2, §3.2.6)
  Cargo.toml, Cargo.lock, src/{lib.rs, parse_line.rs, infer.rs}
deps/build.jl                  # `cargo build --release`, triggered by Pkg.build
py/
  pyproject.toml, uv.lock, logclustering_py/*.py
benchmarks/{loghub2/, regex/}
docs/
  vision.md
  plans/001_upgrade-to-2026-stack.md
```

## Utilities to reuse (do not re-invent)

- `KCompetetive` forward pass (`src/KATE.jl`) — keep the math; the Lux port
  has already wrapped the sort in `@ignore_derivatives`.
- `normalize_log`, `count_words` (`src/KATE.jl`) — reuse as one input-featurisation
  option in `Models/`.
- `get_similar_words` — generalise into `Cluster/Sparsity.jl` for
  decoder-weight cluster interpretation across all models.
- `Clustering.jl`, `Distances.jl`, `UMAP.jl`, `MLUtils.jl`,
  `BytePairEncoding.jl`, `PythonCall.jl` — existing Julia packages.
- `Automa.jl` — regex/FSM → Julia source DFA; pure-Julia fallback in
  `src/Regex/Emit.jl`.
- **Hyperscan** (`libhs`) — multi-pattern SIMD matcher.
- **RE2** — linear-time regex engine; default emit target.
- `xxHash_jll`, `BloomFilters.jl` — streaming dedup in `src/PreProc/Dedup.jl`.

## Global verification gates

1. `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` passes on
   Julia 1.10.
2. `KCompetetive` gradient matches finite differences; Drain round-trip on a
   fixture; PA/GA/FGA/FTA match hand-computed values on a 10-line toy set.
3. End-to-end smoke: `benchmarks/loghub2/run.jl --dataset HDFS --model
   kate_modern` produces a CSV with PA/GA/NMI filled in and no `NaN`s.
4. KATE-modernised reproduces original KATE 20NG NMI within ±2 pts.
5. README benchmark table complete across 14 LogHub-2.0 datasets.
6. LLM closed-loop residual-coverage plateau within ≤ 5 iterations on HDFS.
7. Regex-generation correctness + performance targets met.
8. Pre/post pipeline algebraic properties hold (idempotence, retraction,
   Bloom FPR).

## What is *not* part of this plan

Handled in the vision's "Out of scope" list — streaming ingestion, GUI,
training a log FM from scratch, production hardening.
