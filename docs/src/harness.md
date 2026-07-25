# Harness + CV

Evaluation scaffolding for Stage F. The harness loads LogHub-2.0
CSVs; `CV` generates leak-free train/test splits.

## Harness

```@docs
LogClustering.Harness
LogClustering.Harness.load_loghub
LogClustering.Harness.run_parser
LogClustering.Harness.Dataset
LogClustering.Harness.Report
LogClustering.Harness.format_report
```

## Cross-validation splits

```@docs
LogClustering.CV
LogClustering.CV.Split
LogClustering.CV.time_ordered_split
LogClustering.CV.time_ordered_kfold
LogClustering.CV.per_host_split
LogClustering.CV.per_host_kfold
LogClustering.CV.stratified_by
```
