#!/usr/bin/env julia
# Build a baked Julia sysimage for LogClustering. Run via:
#
#   julia --project=. deploy/build_sysimage.jl /path/to/LogClustering.so
#
# The sysimage embeds the package's precompile workload plus a
# real-CLI run (see precompile_workload.jl) so cold-start `logcluster
# --help` and `logcluster stream --help` both come up in <300 ms.
#
# PackageCompiler is fetched on the fly so it doesn't pollute the
# runtime project — this script only ever runs at image-build time.

using Pkg
try
    @eval using PackageCompiler
catch
    Pkg.add("PackageCompiler"; io = devnull)
    @eval using PackageCompiler
end

if length(ARGS) < 1
    println(stderr, "usage: build_sysimage.jl OUTPUT_PATH")
    exit(2)
end

output = ARGS[1]
here   = @__DIR__
workload = joinpath(here, "precompile_workload.jl")

isfile(workload) ||
    error("precompile workload not found at $workload")

@info "Building LogClustering sysimage" output = output workload = workload

PackageCompiler.create_sysimage(
    [:LogClustering];
    sysimage_path             = output,
    precompile_execution_file = workload,
    incremental               = true,
    cpu_target                = "generic",
)

@info "Done." path = output size_mb = round(filesize(output) / 1024^2; digits = 1)
