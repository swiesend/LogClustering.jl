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
  `py/.venv`) for embedders, UMAP-learn, HDBSCAN, LogPPT — all
  locally-loaded, no cloud-API dependency.

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
  becomes the cluster ID for free. *Ported* in `src/Models/VQVAE.jl`:
  `VectorQuantizer` Lux layer with straight-through estimator,
  `vq_vae` Dense-encoder/decoder Chain, `vq_vae_loss` (reconstruction
  + β·commitment + codebook — equation 3 of the paper),
  `assign_codes` for "cluster id for free".
- **Masked / denoising autoencoder** (Vincent 2008; He et al. MAE 2022) —
  stronger self-supervised signal than pure reconstruction. *Ported*
  in `src/Models/DenoisingAE.jl`: small symmetric Dense encoder /
  decoder, `denoising_ae_loss` applies either a Bernoulli
  mask-to-zero (MAE-style, `mask_rate`) or additive Gaussian
  noise (`σ`) — or both — and reconstructs against the clean input
  with MSE.
- **Contrastive sentence encoder** (SimCSE, Gao et al. 2021) with log-specific
  augmentations (parameter masking, timestamp dropout). *Ported* in
  `src/Models/SimCSE.jl`: `simcse_loss(h, h⁺; τ)` (InfoNCE over cosine
  similarity, deliberately backbone-agnostic) and
  `simcse_step_loss(model, ps, st, x; τ)` (double-forward via
  Dropout RNG). Pairs with any Lux `Chain` with at least one
  Dropout layer.

Replace the regex BoW input with **BPE** via `BytePairEncoding.jl` (or
SentencePiece) trained per-dataset.

**Rationale.** VQ-VAE is the clean 2020s formalisation of what KATE was reaching
for. SimCSE empirically beats reconstruction-trained encoders on clustering.

**Files.** `src/Models/VQVAE.jl` ✓, `src/Models/SimCSE.jl` ✓,
`src/Models/DenoisingAE.jl` ✓; Embedders.jl + Data/Tokenizers.jl
still TODO.

**Verification.** VQ-VAE codebook perplexity stable; SimCSE NMI on HDFS
≥ KATE-modernised + 3 pts.

---

### Stage D — Local-only neural track

**Decision.** Cloud-LLM dependencies (OpenAI, Anthropic, Cohere, Google
Gemini, …) are explicitly **out of scope** — see vision.md's
"Local-only constraint" section. The original plan's LILAC parser and
cluster-labelling / closed-loop LLM tasks are dropped. What remains is
a fully-local neural track:

1. **Embedders** — BGE-m3, GTE, multilingual-E5, Nomic-embed-v1.5,
   Jina-embeddings-v3 served by `sentence-transformers` running in the
   `py/` uv venv. Weights are downloaded once to `~/.cache/huggingface`
   and used offline thereafter. Exposed behind one `Embedder` trait,
   selected by config.
2. **LogPPT parser** — Le & Zhang 2023 — a fine-tuned RoBERTa-base for
   log-template extraction. Local; runs through `transformers`. Sits
   alongside Drain and Brain in the parser comparison.
3. **Optional local-LLM hooks** — if a future feature needs generative
   text (e.g. cluster naming for human consumption), it goes through
   a *local* runtime (Ollama, llama.cpp, MLX) — never a remote API.
   Even those hooks must degrade gracefully: the pipeline produces
   correct cluster ids and templates with the LLM disabled.

**Rationale.** Practitioners need to run the whole pipeline on a
laptop without sending log data — frequently sensitive — to a third
party. Local-only also makes the benchmarks reproducible without
chasing API model-version drift.

**Files (new).** `src/Models/Embedders.jl`, `src/Parsers/LogPPT.jl`,
`py/embedders.py`, `py/logppt.py`.

**Verification.** Embedder NMI on HDFS_2k matches the published
sentence-transformers number within ±1 pt; LogPPT PA matches its
paper number within ±2 pt.

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

**Baselines landed.** `src/Mining/Baselines.jl` adds [`prefixspan`]
(Pei et al. 2001), [`spade`] (Zaki 2001), and [`cmspade`]
(Fournier-Viger et al. 2014), all adapted to single-sequence
serial-episode mining with the same `(sequence; min_sup, max_gap,
max_time_duration)` keyword surface as `mv_span`. Output shape
matches `mv_span`'s (`OrderedDict{Vector{Int}, Vector{Vector{Int}}}`)
so the four miners are directly comparable. They use strictly
non-overlapping minimal occurrences (a more conservative count than
MV-Span's), which means each baseline's pattern *set* may differ
from MV-Span's even on the same input — that difference is
exactly the methodological lever the comparison was meant to
expose.

### Stage E — Clustering & downstream

**Decision.** Standard pipeline for every embedder: L2-normalise → UMAP(n=15,
min_dist=0.0) → HDBSCAN(min_cluster_size=10). Compare k-means and
"clustering-via-sparsity" (KATE hidden-layer argmax) to isolate the thesis's
original contribution.

**Rationale.** McInnes/Healy UMAP + HDBSCAN is the 2024-2026 default;
sparsity-as-clustering lets us ask whether KATE's bottleneck *is* its own
clusterer.

**Sequence-level anomaly detectors landed.** `src/Anomaly/Sequence.jl`
ships three model-agnostic detectors that run against any
`seq_lstm`-shaped predictor (LSTM, peephole, bidirectional):

- `deeplog_topk_anomaly` + `deeplog_scan` — DeepLog (Du et al.
  CCS 2017) top-k membership test + sliding-window variant.
- `masked_surprise` — LogBERT-style (Guo et al. 2021) per-position
  NLL; peaks localise anomalous events inside a sequence.
- `sequence_perplexity` — SeqTransformer-style `exp(mean NLL))` per
  column, the standard Transformer-LM anomaly score.

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

**Post-processing.** Steps 1, 2, 3, 4 are all landed.

1. **Slot-purity re-check** *(landed)* — `src/PostProc/Purity.jl`
   folds low-entropy slots (Shannon entropy < τ, default 0.2 bits)
   back to their literal mode.
2. **Slot typing** *(landed)* — `src/PostProc/Typing.jl` runs the
   anchored typed-regex battery from `PreProc.Masking` over each
   slot's realisations and assigns the most-specific type whose
   match fraction meets `min_match_fraction` (default 0.95).
3. **Template canonicalisation** *(landed)* —
   `src/PostProc/Canonical.jl` sorts alternatives, collapses
   whitespace, unifies placeholder style, and exposes
   `canonical_hash` for cross-parser deduplication.
4. **Cluster merge** *(landed)* — `src/PostProc/Merge.jl` provides
   `token_edit_distance`, `merge_pairs`, and `merge_clusters`:
   canonicalised templates within `max_token_distance` + wildcard
   budget collapse into one group via union-find; surviving
   template is the lex-smallest canonical form.
5. **Regex compilation cache** — hash-indexed compiled `Regex`/
   `RegexSet` DB, shared across evaluation runs. *Partial:* the
   compiled `Rust.RegexSet` multi-pattern matcher exists (Stage E″
   below); the shared cross-run cache layer is TODO.

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
a single, precise, performant regex via the seven-stage ladder below.
Stages 1-7 are all landed (stages 5 uses `RegexSet` as a local,
library-free substitute for Hyperscan — swap is a one-file change
behind the same Julia API).

1. **Anti-unification over aligned tokens** *(landed)* — token-level
   greedy LCS across deduplicated cluster representatives. Exposed
   via `Rust.infer_regex(samples; align = true)`.
2. **Per-slot regex inference (MDL ladder)** *(landed)* —
   `Rust.slot_ladder(values; typed, enum_max)` picks the tightest
   regex from exact-literal → enum `(?:a|b|c)` → typed class (via
   `PreProc.Masking`'s ranked battery) → bounded `\d{m,n}` /
   `\w{m,n}` / `[A-Za-z0-9]{m,n}` → unbounded wildcard.
3. **Alternation minimisation** *(landed)* — `Rust.alt_min(alts)`
   dedups + sorts + escapes a list of alternatives into a canonical
   `(?:a|b|c)` group.
4. **Emit RE2** *(landed)* — every `Rust.*` emitter is already
   RE2-compatible (Rust `regex` crate is a direct RE2 descendant).
5. **Multi-pattern hot path** *(landed as `RegexSet`; Hyperscan
   optional)* — `Rust.RegexSet([patterns...])` compiles the whole
   set into a single linear-time matcher; `regexset_match(set, line)`
   returns the 1-based indices of every pattern that matched. This
   substitutes for Hyperscan on systems without `libhs`. A
   Hyperscan-backed variant is a one-file swap behind the same
   Julia API.
6. **Verification pass** *(landed)* — `Rust.verify_pattern(pattern,
   samples) -> (hits, total)` gives anchored coverage counts over
   a cluster.
7. **Cost monitoring** *(landed)* — `Rust.mdl_cost(pattern, samples)`
   returns a two-part MDL-style bit cost (pattern length × 7 + Σ
   residual × 7). Monotone but rough; enough to rank candidates so
   the ladder picks tight over loose at equal coverage.

**LLM assist (optional, local-only).** After deterministic generation,
optionally call out to a *local* LLM (Ollama / llama.cpp) for a tighter
slot class; accept only if the candidate matches 100 % of the cluster
and stays RE2-compatible. LLM proposes, deterministic verifier decides.
The whole pipeline must work with the LLM hook disabled — see
vision.md's "Local-only constraint" section.

**Rationale.** RE2/Hyperscan give worst-case linear matching. Hyperscan scans
the entire template set in one pass — throughput scales with *characters*, not
templates. MoLFI-style evolutionary search is explicitly rejected: slow and
unnecessary once anti-unification + typed slots are in place.

**Files.** Stage E″ lives in Rust (`rust/src/mdl.rs`,
`rust/src/infer.rs`) behind Julia wrappers in `src/Rust.jl`
(`slot_ladder`, `alt_min`, `verify_pattern`, `mdl_cost`, `RegexSet`,
`regexset_match`, `infer_regex`). The originally-planned
`src/Regex/*.jl` tree collapsed into those bindings — the
"Rust-does-the-hot-loop, Julia-does-the-orchestration" split from
plan 002 applies here too.

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

**Baselines table** (in `README.md`): Drain, Brain, LogPPT,
BGE+HDBSCAN (local sentence-transformers), KATE-original,
KATE-modernised, VQ-VAE, SimCSE-logs across all 14 datasets. LILAC
and other API-bound parsers are explicitly excluded — see
vision.md's local-only constraint.

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
  Parsers/{Drain.jl, Brain.jl, LogPPT.jl}
  Models/{VQVAE.jl, DenoisingAE.jl, SimCSE.jl, Embedders.jl,
          SeqLSTM.jl (ported, thesis §3.2.8), SeqTransformer.jl, SeqMamba.jl}
  Cluster/{Pipeline.jl, Sparsity.jl}
  Mining/{Episodes.jl,              # MV-Span / MT-Span (thesis §3.2.7, ported)
          Baselines.jl}             # PrefixSpan, SPADE, CM-SPADE (Stage E baselines)
  Anomaly/{Instance.jl (ported, thesis §3.2.5), Sequence.jl}
  RCA/{Graph.jl, CausalDiscovery.jl}
  Eval/{Metrics.jl, Harness.jl, CV.jl}
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
6. Regex-generation correctness + performance targets met.
7. Pre/post pipeline algebraic properties hold (idempotence, retraction,
   Bloom FPR).
8. Every gate passes with the network detached after the initial weight
   download — see vision.md's "Local-only constraint".

## What is *not* part of this plan

Handled in the vision's "Out of scope" list — streaming ingestion, GUI,
training a log FM from scratch, production hardening, **cloud-LLM
dependencies** (every neural step runs against locally-cached
weights only).
