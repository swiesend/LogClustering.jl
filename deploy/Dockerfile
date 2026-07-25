# syntax=docker/dockerfile:1.6
# ----------------------------------------------------------------------
# LogClustering — production container.
#
# Two stages:
#  1. `builder` resolves the project into a self-contained depot,
#     precompiles every transitive package, and bakes a custom Julia
#     sysimage via PackageCompiler so cold-start `logcluster --help`
#     runs in <300 ms instead of the usual ~6 s.
#  2. `runtime` carries the sysimage, the project tree (read-only),
#     the shell shim, AND the Julia depot — the jll artifacts
#     (SQLite, TLS libraries, HDF5 for JLD2) are dlopen'd from the
#     depot at runtime, so it must ship with the image.
#
# The runtime base is `julia:1.10` (NOT `julia:1.10-slim` — the
# official Julia images publish no `-slim` variant).
# ----------------------------------------------------------------------

ARG JULIA_VERSION=1.10

# ====================  Stage 1: builder  ===============================
FROM julia:${JULIA_VERSION} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pin the depot to a known path so the runtime stage can copy it.
ENV JULIA_DEPOT_PATH=/opt/julia-depot
WORKDIR /build

# Project files first — keeps Docker layer caching effective when only
# src/ changes.
COPY Project.toml Manifest.toml ./
COPY src/        ./src/
COPY rust/       ./rust/
COPY bin/        ./bin/
COPY deploy/build_sysimage.jl      deploy/build_sysimage.jl
COPY deploy/precompile_workload.jl deploy/precompile_workload.jl

# Instantiate + precompile the whole dep tree into the pinned depot.
# `JULIA_NUM_PRECOMPILE_TASKS` defaults to ncpu — set to 4 here to keep
# memory under 4 GB on smaller CI runners.
ENV JULIA_NUM_PRECOMPILE_TASKS=4
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Build the sysimage. Output lands at /build/LogClustering.so.
RUN julia --project=. deploy/build_sysimage.jl /build/LogClustering.so

# ====================  Stage 2: runtime  ===============================
FROM julia:${JULIA_VERSION} AS runtime

LABEL org.opencontainers.image.source="https://github.com/swiesend/LogClustering.jl"
LABEL org.opencontainers.image.description="LogClustering streaming inference + alerting"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user. UID 65532 is the conventional `nonroot` uid.
RUN groupadd -g 65532 logcluster && \
    useradd  -u 65532 -g logcluster -m -s /sbin/nologin logcluster && \
    mkdir -p /var/lib/logcluster /etc/logcluster && \
    chown logcluster:logcluster /var/lib/logcluster

WORKDIR /opt/logclustering

# Julia depot (packages + artifacts + precompiled cache). Without this
# the SQLite / TLS / HDF5 jll shared libraries can't be dlopen'd and
# `stream --memory` / webhook TLS fail at first use.
COPY --from=builder /opt/julia-depot /opt/julia-depot

# Project tree (read-only at runtime).
COPY --from=builder /build/Project.toml ./Project.toml
COPY --from=builder /build/Manifest.toml ./Manifest.toml
COPY --from=builder /build/src/        ./src/
COPY --from=builder /build/bin/        ./bin/
COPY --from=builder /build/rust/       ./rust/
# Sysimage.
COPY --from=builder /build/LogClustering.so /opt/logclustering/LogClustering.so

ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV JULIA_PROJECT=/opt/logclustering
ENV JULIA_SYSIMAGE_PATH=/opt/logclustering/LogClustering.so
ENV PATH=/opt/logclustering/bin:$PATH

# SIGINT (not the default SIGTERM) so the stream subcommand's signal
# handler runs its graceful drain on `docker stop`.
STOPSIGNAL SIGINT

USER logcluster:logcluster
ENTRYPOINT ["/opt/logclustering/bin/logcluster"]
CMD ["--help"]
