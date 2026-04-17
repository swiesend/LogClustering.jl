# Vision — LogClustering.jl

## Origin

`LogClustering.jl` is a 2018 Julia thesis artefact: a port of the KATE paper
(Chen & Zaki, *KATE: K-Competitive Autoencoder for Text*, KDD 2017). The repo
carried the *idea* of the thesis but almost none of the pipeline its name
promises. This document captures the long-term direction; concrete, staged
decisions live in [`plans/`](plans/).

## Principle ideas from the thesis (preserve these)

1. **Sparse competitive bottleneck** as a clustering prior — forcing only `k`
   neurons to fire per input pushes semantically similar logs to share active
   neurons, so neurons behave like cluster indicators.
2. **Log-count normalisation** as input featurisation — treats token
   frequencies as noisy signals and dampens high-frequency tokens.
3. **Word-embedding interpretability via decoder weights** — cluster meaning
   is inspectable by projecting back into the vocabulary.
4. **Unsupervised reconstruction** as the single training signal — no labels
   required, suitable for operational log streams.

Every future extension must keep these four ideas intelligible and falsifiable.

## Gaps the thesis left open

| Gap | Consequence |
|---|---|
| No clustering step on learned embeddings | "LogClustering" name is aspirational. |
| No log dataset loader | Evaluated on Shakespeare; log-specific behaviour unverified. |
| No evaluation metrics | Cannot quantify whether KATE helps. |
| No baselines | Impossible to position against Drain / Brain / LogPPT (all locally-runnable). |
| No log-aware tokeniser | Regex splits break numeric/IP/path tokens. |
| Obsolete Flux API (Julia 0.6) | Project could not run on current Julia. |

## The original thesis pipeline

The thesis figure lays out a pipeline far larger than what the repo originally
implemented. Only the **Instanzbasiertes Modell (Autoencoder)** node existed in
code; every other node was intended. The vision is to preserve the figure's
topology and upgrade each node to its modern counterpart. Dashed edges
(validation feedback, CV feedback) are preserved as evaluation loops.

```
Log ─► Event-Log Parsing ─► Event-Log ─► Event-Log Corpus
                               │                 │
                               ├─► Autoencoder   └─► Log-Key Inference
                               │      │                   │
                               └─► Dim-Reduction          │
                                      │                   │
                                      └─► Clustering ◄────┤
                                           │       ▲      │
                                           │       └─► Clustering Validation
                                           │                 │
                                           ▼                 ▼
                 Cross-Val ◄──► Sequential Model (LSTM) ◄─ Episode Mining
                                           │                 │
                          ┌────────────────┤                 │
                          ▼                ▼                 ▼
          Outlier (Instance)   Outlier (Sequence)   Root-Cause Analysis
```

## Node-by-node modernisation

| Thesis node (figure, German) | What it meant | Modern counterpart |
|---|---|---|
| **Event-Log Parsing** | Turn raw lines into structured events | Typed-slot pre-processing → Drain3 → LogPPT (local fine-tuned RoBERTa) → generic regex emission |
| **Event-Log** | Sequence of structured events with slots | `EventSchema` (template-id, slot values, timestamp, host) |
| **Event-Log Corpus** | Persistent, query-able corpus | Parquet + DuckDB; Arrow columnar; indexed by template-id and time |
| **Log-Key Inferenz** | Stable IDs per distinct template | Canonical-template hash → stable integer key |
| **Instanzbasiertes Modell (Autoencoder)** | KATE | KATE-modernised + VQ-VAE + denoising AE + SimCSE + locally-loadable HuggingFace encoders (BGE/E5/Nomic/Jina via downloaded weights) behind a common `Embedder` interface |
| **Dimensionsreduktion (Einbettung)** | Project to low-D space | UMAP (McInnes 2018), PaCMAP (Wang 2021); ivis for supervised DR |
| **Clustering** | Group similar events | HDBSCAN default; k-means + sparsity-based (KATE argmax) as comparators |
| **Clustering Validierung** | Validate assignment against log-keys | PA/GA/FGA/FTA/NMI/ARI/homogeneity; iterative re-embedding on low-purity clusters |
| **Episode Mining** | Frequent sequential patterns | Thesis's MV-Span (SPADE-derived, projected vertical DB) + MT-Span (TSpan/EWU/IESC) ported in `src/Mining/Episodes.jl`; PrefixSpan (Pei 2001), SPADE (Zaki 2001), CM-SPADE; Transformer attention-motif mining |
| **Sequentielles Modell (LSTM)** | Predict next event | Transformer decoder (DeepLog → LogBERT lineage); Mamba/SSM (Gu & Dao 2024). LSTM baseline ported from the thesis in `src/Models/SeqLSTM.jl` (Lux `Recurrence(LSTMCell)` + `Embedding` + `Dense`; bi-directional + peephole variants TODO) |
| **Kreuzvalidierung** | k-fold CV | Time-ordered / per-host splits to avoid leakage |
| **Ausreißererkennung (Instanz)** | Detect anomalous single lines | AE/VQ-VAE reconstruction error; HDBSCAN outlier score; isolation forest; value-novelty over masked slots |
| **Ausreißererkennung (Sequenz)** | Detect anomalous sequences | DeepLog top-k next-event; LogBERT masked-event surprise; SeqTransformer perplexity |
| **Root-Cause-Analysis** | Identify causal chain | Episode-graph causal discovery (PC, LiNGAM, PCMCI+); counterfactual ablation |

## End-to-end topology

```
raw log stream
  │
  ▼
[PreProc]  Framing → Typed-Slot Masking → BPE → Stream Dedup
  │
  ▼
[Parsing]  Drain3 ───► Log-Key Inference  ─────┐
  │      └─► LogPPT (local RoBERTa, residuals) │
  │                                            │
  ▼                                            │
[Regex] AntiUnify → SlotLadder → RE2/Hyperscan │
  │                                            │
  ▼                                            ▼
[Embed] KATE / VQ-VAE / SimCSE / BGE-local ◄─── Event-Log Corpus (Parquet + DuckDB)
  │
  ▼
[DR]    UMAP / PaCMAP
  │
  ▼
[Cluster] HDBSCAN ─► Sparsity-Cluster ─► Validation ◄── Log-Keys
  │                                         │
  ▼                                         ▼ (dashed feedback)
[Sequence] Transformer / Mamba / LSTM  ◄── Episode Mining
  │                   │                    │
  ▼                   ▼                    ▼
Instance         Sequence-Anomaly    Root-Cause Analysis
Anomaly          (DeepLog/LogBERT)   (PC / LiNGAM / counterfactual)
```

This is the figure, extended — not replaced. Every original arrow survives;
every original box has a modern interior.

## Local-only constraint

`LogClustering.jl` runs offline. The pipeline must work end-to-end on a
laptop with no public-internet calls beyond the initial download of
model weights / datasets — no OpenAI, Anthropic, Cohere, or other
remote-LLM-API dependency. Anywhere a "neural inference" step appears,
it's served by:

- **Locally-loaded HuggingFace weights** — `transformers` /
  `sentence-transformers` running through the `py/` uv venv (BGE,
  GTE, E5, Nomic, Jina, RoBERTa-based LogPPT).
- **Pure-Julia models** trained in this repo (KATE, DeepKATE, VQ-VAE,
  SimCSE, SeqLSTM, peephole-LSTM).
- **Native algorithms** (Drain, MV-Span, MT-Span, k-means, HDBSCAN
  via Python).

If a future feature genuinely needs a generative LLM, it goes through
a *local* runtime (Ollama, llama.cpp, MLX) — never a remote API.

## Theoretical spine

The compression / encoder-decoder track is framed in rate–distortion and MDL
terms (Grünwald 2007): logs compress ~10× with gzip, so **clustering quality
≈ description-length reduction**. Reported BPC of VQ-VAE codes vs. gzip vs.
Drain's dictionary gives the thesis the coherent spine it previously lacked.

## Language strategy

Julia stays the core (preserves thesis identity and the `KCompetetive` layer).
A Python companion handles the HuggingFace-only pieces (embedders, UMAP-learn,
HDBSCAN, LogPPT) via `PythonCall.jl`. A full Python rewrite would erase the
thesis; a pure-Julia rewrite would re-implement two years of HuggingFace work.
The Python side ships through the `py/` uv venv and downloads weights once
to a local cache — no runtime API calls.

## Out of scope

- Real-time streaming ingestion (Kafka / Vector / OTEL).
- A user-facing GUI or dashboard.
- Training a log-specific foundation model from scratch — we only fine-tune
  and prompt existing ones.
- Production-hardening (auth, multi-tenancy, storage).
- **Cloud-LLM dependencies** (OpenAI, Anthropic, Cohere, Google Gemini, etc.).
  Every feature must run offline once weights are cached. LILAC-style
  prompted parsers are explicitly out — LogPPT covers the same niche
  with a fully-local fine-tuned RoBERTa.

These are natural follow-ons but dilute the thesis-extension narrative.

## Further reading

- [`plans/001_upgrade-to-2026-stack.md`](plans/001_upgrade-to-2026-stack.md)
  — the concrete staged plan with decisions, modules, and verification.
- [`plans/002_persistence-autotune-cli.md`](plans/002_persistence-autotune-cli.md)
  — persistence / auto-tune / CLI: make every trained artifact
  reproducible, right-sized from data, and usable from a shell.
