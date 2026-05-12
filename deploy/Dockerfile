# syntax=docker/dockerfile:1.6
# ----------------------------------------------------------------------
# LogClustering — production container.
#
# Two stages:
#  1. `builder` resolves the project, precompiles every transitive
#     package, and bakes a custom Julia sysimage via PackageCompiler so
#     cold-start `logcluster --help` runs in <300 ms instead of the
#     usual ~6 s. Build is heavy (~5 min on a 4-core box, ~700 MB
#     image) and only happens at image-build time.
#  2. `runtime` is a slim Julia base that carries only the sysimage,
#     the project tree (read-only), and the shell shim.
# ----------------------------------------------------------------------

ARG JULIA_VERSION=1.10

# ====================  Stage 1: builder  ===============================
FROM julia:${JULIA_VERSION} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Project files first — keeps Docker layer caching effective when only
# src/ changes.
COPY Project.toml Manifest.toml ./
COPY src/        ./src/
COPY rust/       ./rust/
COPY bin/        ./bin/
COPY deploy/build_sysimage.jl      deploy/build_sysimage.jl
COPY deploy/precompile_workload.jl deploy/precompile_workload.jl

# Instantiate + precompile the whole dep tree. `JULIA_NUM_PRECOMPILE_TASKS`
# defaults to ncpu — set to 4 here to keep memory under 4 GB on smaller
# CI runners.
ENV JULIA_NUM_PRECOMPILE_TASKS=4
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Build the sysimage. Output lands at /build/LogClustering.so.
RUN julia --project=. deploy/build_sysimage.jl /build/LogClustering.so

# ====================  Stage 2: runtime  ===============================
FROM julia:${JULIA_VERSION}-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/swiesend/LogClustering.jl"
LABEL org.opencontainers.image.description="LogClustering streaming inference + alerting"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user. UID 65532 is the conventional `nonroot` uid.
RUN groupadd -g 65532 logcluster && \
    useradd  -u 65532 -g logcluster -m -s /sbin/nologin logcluster && \
    mkdir -p /var/lib/logcluster /etc/logcluster && \
    chown logcluster:logcluster /var/lib/logcluster

WORKDIR /opt/logclustering

# Project tree (read-only at runtime).
COPY --from=builder /build/Project.toml ./Project.toml
COPY --from=builder /build/Manifest.toml ./Manifest.toml
COPY --from=builder /build/src/        ./src/
COPY --from=builder /build/bin/        ./bin/
COPY --from=builder /build/rust/       ./rust/
# Sysimage.
COPY --from=builder /build/LogClustering.so /opt/logclustering/LogClustering.so

ENV JULIA_PROJECT=/opt/logclustering
ENV JULIA_SYSIMAGE_PATH=/opt/logclustering/LogClustering.so
ENV PATH=/opt/logclustering/bin:$PATH

USER logcluster:logcluster
ENTRYPOINT ["/opt/logclustering/bin/logcluster"]
CMD ["--help"]
