#
# Build the companion Rust crate at `rust/`.
#
# Invoked by `Pkg.build("LogClustering")`. Produces
# `rust/target/release/liblogclustering_rs.{so,dylib,dll}`; the Julia
# module `LogClustering.Rust` locates it via `Libdl.find_library`.
#
# Requires a working `cargo` on PATH; on CI the job installs Rust via
# `dtolnay/rust-toolchain@stable` before `julia-buildpkg`.

const ROOT = abspath(joinpath(@__DIR__, ".."))
const CRATE = joinpath(ROOT, "rust")

function have_cargo()
    try
        return success(pipeline(`cargo --version`, stdout = devnull, stderr = devnull))
    catch
        return false
    end
end

if !have_cargo()
    @warn "cargo not found on PATH — skipping Rust build. \
           Install Rust (https://rustup.rs) and re-run Pkg.build(\"LogClustering\") \
           to enable the Rust-accelerated kernels."
    exit(0)
end

@info "Building logclustering_rs (release)" crate = CRATE
run(Cmd(`cargo build --release`; dir = CRATE))
@info "logclustering_rs built"
