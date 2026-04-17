# Rust FFI

Performance-critical kernels live in `rust/logclustering_rs`.
`deps/build.jl` calls `cargo build --release` on install; the
`LogClustering.Rust` Julia module discovers the shared library via
`Libdl` and calls it through `@ccall`.

## parse_line (thesis Algorithm 3.1)

Recursive regex-cascade over a single line.

```@docs
LogClustering.Rust.parse_line
LogClustering.Rust.Span
```

## infer_regex (thesis Algorithm 3.10 + anti-unification)

Log-key regex inference. `align = true` turns on LCS-based
anti-unification for variable-length samples.

```@docs
LogClustering.Rust.infer_regex
LogClustering.Rust.abi_version
LogClustering.Rust.ABI_VERSION
```
