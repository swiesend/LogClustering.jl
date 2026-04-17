# Clustering

Stage E. The pipeline is `L2-normalise → [UMAP] → {kmeans, HDBSCAN,
sparsity}`. UMAP + HDBSCAN load Python lazily via `PythonCall`; the
native paths stay Julia-only.

## Sparsity clustering (KATE argmax)

```@docs
LogClustering.Sparsity
LogClustering.Sparsity.sparsity_clusters
LogClustering.Sparsity.sparsity_labels
LogClustering.Sparsity.top_k_indices
```

## Distance-based pipeline

```@docs
LogClustering.Pipeline
LogClustering.Pipeline.l2_normalise
LogClustering.Pipeline.l2_normalise!
LogClustering.Pipeline.kmeans_cluster
LogClustering.Pipeline.umap_reduce
LogClustering.Pipeline.hdbscan_cluster
LogClustering.Pipeline.umap_hdbscan
```
