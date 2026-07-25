# Parsers & post-processing

Stage B + Stage E′ steps 1–4.

## Drain3

```@docs
LogClustering.Drain3
LogClustering.Drain3.Drain
LogClustering.Drain3.process!
LogClustering.Drain3.parse_all
LogClustering.Drain3.LogCluster
LogClustering.Drain3.template_of
```

## PostProc — after-parse tightening (Stage E′)

Three deterministic refinements that run on a parser's templates +
per-cluster realisations. Order: Canonical → Purity → Typing →
Merge.

### Canonical

```@docs
LogClustering.Canonical
LogClustering.Canonical.canonicalise
LogClustering.Canonical.canonicalise_all
LogClustering.Canonical.canonical_hash
```

### Purity — entropy-based slot promotion

```@docs
LogClustering.Purity
LogClustering.Purity.slot_entropy
LogClustering.Purity.slot_realisations
LogClustering.Purity.promote_pure_slots
```

### Typing — promote `<*>` to the most-specific typed class

```@docs
LogClustering.Typing
LogClustering.Typing.type_slot
LogClustering.Typing.type_template
```

### Merge — collapse sibling templates

```@docs
LogClustering.Merge
LogClustering.Merge.token_edit_distance
LogClustering.Merge.merge_pairs
LogClustering.Merge.merge_clusters
```
