# Rust FFI

Performance-critical kernels live in `rust/logclustering_rs`.
`deps/build.jl` calls `cargo build --release` on install; the
`LogClustering.Rust` Julia module discovers the shared library via
`Libdl` and calls it through `@ccall`.

## parse_line / parse_lines (thesis Algorithm 3.1)

Recursive regex-cascade over a single line, and the batched form
that amortises one FFI crossing per corpus — 400–1000× faster
than the per-line loop on 2 k-line LogHub-2.0 subsets.

```@docs
LogClustering.Rust.parse_line
LogClustering.Rust.parse_lines
LogClustering.Rust.Span
```

## infer_regex (thesis Algorithm 3.10 + anti-unification)

Log-key regex inference. `align = true` turns on LCS-based
anti-unification for variable-length samples.

```@docs
LogClustering.Rust.infer_regex
```

## Stage E″ MDL ladder

`slot_ladder` / `alt_min` / `verify_pattern` / `mdl_cost`
implement the five-rung ladder (exact-literal → enum → typed class
→ bounded shape → wildcard) plus a sort/dedup/escape minimiser, a
pattern coverage check, and a two-part bits-per-sample MDL score.

```@docs
LogClustering.Rust.slot_ladder
LogClustering.Rust.alt_min
LogClustering.Rust.verify_pattern
LogClustering.Rust.mdl_cost
```

## RegexSet (Hyperscan substitute)

`regex::RegexSet` wrapped as a Julia handle with a finalizer.
Compiles N patterns into one RE2-flavoured linear-time matcher and
returns the 1-based indices of every pattern that matched a line.
Drop-in replacement for Hyperscan on systems without `libhs`; a
Hyperscan-backed variant is a one-file swap behind this API.

```@docs
LogClustering.Rust.RegexSet
LogClustering.Rust.regexset_match
```

## Versioning

```@docs
LogClustering.Rust.abi_version
LogClustering.Rust.ABI_VERSION
```
