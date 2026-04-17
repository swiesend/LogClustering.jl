# Compression

The plan's MDL spine: rate every template dictionary against gzip and
measure codebook usage.

```@docs
LogClustering.Compression
LogClustering.Compression.dictionary_size
LogClustering.Compression.bpc_gzip
LogClustering.Compression.bpc_dictionary
LogClustering.Compression.codebook_perplexity
LogClustering.Compression.entropy_bits
LogClustering.Compression.entropy_bits_from_counts
```
