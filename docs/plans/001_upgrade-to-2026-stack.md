# Plan 001 — Upgrade to the 2026 stack

| Field | Value |
|---|---|
| Status | In progress (Stage A partially landed) |
| Scope | Revive `LogClustering.jl` and realise the full thesis pipeline on modern tooling |
| Supersedes | — |
| See also | [`../vision.md`](../vision.md) |

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
- [x] Wire a `py/` companion project (pyproject.toml + `PythonCall.jl` +
  `CondaPkg.toml`) for embedders, UMAP-learn, HDBSCAN, LILAC.

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

**Decision.** Keep KATE-modernised as the reference. Add three siblings:

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

### Stage E — Clustering & downstream

**Decision.** Standard pipeline for every embedder: L2-normalise → UMAP(n=15,
min_dist=0.0) → HDBSCAN(min_cluster_size=10). Compare k-means and
"clustering-via-sparsity" (KATE hidden-layer argmax) to isolate the thesis's
original contribution.

**Rationale.** McInnes/Healy UMAP + HDBSCAN is the 2024-2026 default;
sparsity-as-clustering lets us ask whether KATE's bottleneck *is* its own
clusterer.

**Files (new).** `src/Cluster/{Pipeline.jl, Sparsity.jl}`,
`src/Anomaly/{Instance.jl, Sequence.jl}`.

**Verification.** Three clustering methods produce comparable NMI on HDFS
(within 5 pts).

---

### Stage E′ — Pre- and post-processing pipeline

**Decision.** Replace `transform_text_to_input` (a single regex split plus
log-count normalisation) with a layered, reorderable pipeline.

**Pre-processing.**
1. **Structural framing** — strip collector prefix (syslog RFC-5424,
   journald, Kubernetes CRI, Docker JSON-lines). Grammar registry in
   `PreProc/Framing.jl`.
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

### Stage F — Evaluation

**Decision.** LogHub-2.0 (Zhu et al. ISSRE 2023) as the single source of
ground truth — all 14 datasets, ~50 k annotated lines each.

**Metrics.**
- Parsing: PA, GA, FGA, FTA (Zhu et al. 2023, current field standard).
- Clustering: NMI, ARI, Purity, V-measure.
- Compression: BPC of codes, codebook perplexity, dictionary size.

**Baselines table** (in `README.md`): Drain, Brain, LogPPT, LILAC,
BGE+HDBSCAN, KATE-original, KATE-modernised, VQ-VAE, SimCSE-logs across all
14 datasets.

**Reproduction gate.** KATE-modernised matches the original KATE paper's
20NG clustering NMI within ±2 pts as a sanity check.

**Files (new).** `src/Eval/{Metrics.jl, Harness.jl, CV.jl}`,
`benchmarks/loghub2/`.

---

## Module map after this plan

```
src/
  KATE.jl                      # modernised (Stage A, done)
  LogClustering.jl             # umbrella
  Data/{LogHub.jl, Tokenizers.jl, EventLog.jl, Corpus.jl}
  PreProc/{Framing.jl, Masking.jl, Dedup.jl}
  PostProc/{Purity.jl, Typing.jl, Canonical.jl, Merge.jl}
  Regex/{AntiUnify.jl, SlotLadder.jl, Minimize.jl, Emit.jl,
         Hyperscan.jl, Verify.jl}
  Parsers/{Drain.jl, Brain.jl, LogPPT.jl, LILAC.jl}
  Models/{VQVAE.jl, DenoisingAE.jl, SimCSE.jl, Embedders.jl,
          SeqLSTM.jl, SeqTransformer.jl, SeqMamba.jl}
  Cluster/{Pipeline.jl, Sparsity.jl}
  Mining/{Episodes.jl}
  Anomaly/{Instance.jl, Sequence.jl}
  RCA/{Graph.jl, CausalDiscovery.jl, LLM.jl}
  Eval/{Metrics.jl, Harness.jl, CV.jl}
  LLM/{Loop.jl, Labeling.jl}
py/
  pyproject.toml, embedders.py, lilac_parser.py, umap_hdbscan.py
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
