# Root-Cause-Analysis (RCA) pipeline

The `rca` subcommand and the underlying [`LogClustering.RCA`](@ref)
module compose the Stage C embedder, Stage E anomaly detectors,
and the Stage E episode miners into a single end-to-end
"lines → root-cause report" workflow. No new kernels are added —
RCA is pure orchestration over pieces this package already ships.

## What RCA answers

*"Which recurring sequences of events correlate with anomalous
lines?"*

Given a trained DeepKATE (or VQ-VAE) autoencoder and — optionally —
a trained [`ValueNoveltyDetector`](@ref), RCA returns:

1. A **cluster id per line** (from k-means on the L2-normalised
   latent).
2. A **per-line anomaly score** fusing template reconstruction
   error with value novelty.
3. A set of **serial episodes** mined from the cluster-id stream,
   seeded with the cluster ids that appear in the top-percentile
   most anomalous lines.
4. A **ranked list of root-cause candidates** scored by
   `support × mean_anomaly_density` — patterns that (a) happen
   often and (b) happen in anomalous windows.

## Typical workflow

```sh
# 1. (once) Train an encoder on the target corpus. `--auto` picks
#    `hidden` + `latent` from a PCA elbow; `--reuse` + `--registry`
#    skip SGD entirely if a compatible bundle already exists.
logcluster train --kind deep_kate \
    --data /var/log/prod.log \
    --auto --budget 10 \
    --reuse --registry ~/.cache/logclustering/models \
    --out /var/tmp/prod-dk.jld2

# 2. (once) Train a value-novelty detector — one update per clean line.
#    (Wrap with your own script for now; a `fit-detector` CLI stub
#    is tracked in plan 001 as follow-up.)

# 3. (per investigation) Run the RCA pipeline.
logcluster rca \
    --model    /var/tmp/prod-dk.jld2 \
    --detector /var/tmp/prod-vnd.jld2 \
    --data     /var/log/prod.log \
    --format   md \
    --topk     10 \
    --percentile 0.05 \
    --min-sup 3 --max-gap 20 --max-dur 30 \
    --out      /var/tmp/rca.md

head -40 /var/tmp/rca.md
```

## Knobs worth knowing

- **`--percentile`** (default `0.10`) — the anomaly cutoff that
  decides which cluster ids seed the episode miner. Tighter values
  (`0.02`) zoom in on the rarest events; looser values (`0.25`)
  catch broad drift.
- **`--min-sup`**, **`--max-gap`**, **`--max-dur`** — standard
  [`mv_span`](@ref) constraints. Keep `max-dur` modest
  (< corpus length / 20 is a good rule of thumb) — the miner's
  runtime grows with the pattern lattice size.
- **`--k-clusters`** — override the
  `max(4, ceil(sqrt(n_lines)))` default when you know how many
  structural event templates you expect.

## Output shape

```markdown
# RCA report

- lines: **2 000**
- clusters: **45**
- anomaly cutoff (percentile 0.05): **0.7231**
- anomalous lines: **100**
- episode knobs: min_sup=3, max_gap=20, max_time_duration=30
- detector used: true

## Top 5 root-cause episodes

| # | pattern (cluster-id seq) | support | density | score |
|---|---|---:|---:|---:|
| 1 | `[7, 13]`       | 27 | 0.7881 | 21.28 |
| 2 | `[7, 13, 13]`   | 19 | 0.8014 | 15.23 |
| 3 | `[4]`           | 38 | 0.3602 | 13.69 |
| …  |                |    |        |       |

## Representative lines (first occurrence of each of top 5)

### #1 pattern `[7, 13]`
- L412: `ERROR db connection lost (pool=2/8)`
- L413: `ERROR db connection lost (pool=3/8)`
…
```

Integrate the Markdown block into your incident-review runbook
and iterate on the percentile + min-sup knobs until the top-N
episodes reliably isolate the failure mode.

## Julia API

```@docs
LogClustering.RCA
LogClustering.RCA.root_cause
LogClustering.RCA.render_markdown
LogClustering.RCA.RCAReport
```

## Model reuse across investigations

RCA reads any DeepKATE bundle, so the usual pattern is to train
once per environment and keep the bundle in the registry. The
[`train --reuse`](@ref) flag + [`select-model`](@ref) subcommand
use a SHA-256 [`corpus_fingerprint`](@ref) of the sorted
vocabulary to decide whether an existing bundle can be reused on
a new corpus — if the vocabulary matches, SGD is skipped and the
saved bundle is copied to `--out`.
