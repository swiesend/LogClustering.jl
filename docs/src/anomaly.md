# Anomaly

Thesis §3.2.5 + plan 001 Stage E.

## Instance-level — reconstruction error on any Lux AE

```@docs
LogClustering.Instance
LogClustering.Instance.reconstruction_error_abs
LogClustering.Instance.reconstruction_error_sq
LogClustering.Instance.latent_distance
LogClustering.Instance.anomaly_score
```

## Value-novelty — IP / user / number out-of-distribution scoring

```@docs
LogClustering.Instance.ValueNoveltyDetector
LogClustering.Instance.update!
LogClustering.Instance.value_novelty
LogClustering.Instance.combined_anomaly
```

## Sequence-level — DeepLog + LogBERT + perplexity

Plan 001 Stage E. Model-agnostic detectors that run against any
`seq_lstm`-shaped predictor (`(T, B) → (vocab, B)` logits).

```@docs
LogClustering.Sequence
LogClustering.Sequence.deeplog_topk_anomaly
LogClustering.Sequence.deeplog_scan
LogClustering.Sequence.masked_surprise
LogClustering.Sequence.sequence_perplexity
```
