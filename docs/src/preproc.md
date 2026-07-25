# Pre-processing

Plan 001 Stage E′. Run order: [Framing](@ref) → [Masking](@ref) →
[Dedup](@ref). Every layer is optional — each component preserves
its input's byte-level meaning (Framing uses zero-copy `SubString`
views; Masking restores original runs between matched spans; Dedup
is observational only).

## Framing

```@docs
LogClustering.Framing
LogClustering.Framing.parse_frame
LogClustering.Framing.Frame
LogClustering.Framing.foreach_sd_element
LogClustering.Framing.foreach_sd_param
```

## Masking

```@docs
LogClustering.Masking
LogClustering.Masking.mask_line
LogClustering.Masking.mask_lines
LogClustering.Masking.DEFAULT_LABELS
LogClustering.Masking.DEFAULT_PATTERNS
```

## Dedup

```@docs
LogClustering.Dedup
LogClustering.Dedup.DedupState
LogClustering.Dedup.is_new!
LogClustering.Dedup.contains
LogClustering.Dedup.dedup
LogClustering.Dedup.fpr_estimate
```
