"""
    Stats.TDigest

Minimal, pure-Julia [t-digest](https://arxiv.org/abs/1902.04023) for
streaming quantile estimation. Used by the rules engine's `auto:p99`
/ `auto:zscore` thresholds so per-line threshold evaluation is
O(log δ) amortized instead of the old sort-per-fire over a
1024-element FIFO — and, unlike that FIFO, the estimate reflects the
whole stream (no recency bias) while staying bounded in memory.

The digest is a set of *centroids* `(mean, weight)` kept sorted by
mean. Buffered singletons are merged in when the buffer fills, and
the merged centroid list is compressed back under `δ` (compression)
using the standard `k1` scale function so centroids near the tails
(where quantile accuracy matters for alerting) stay small.

## Surface

- [`TDigest`] — the sketch. Construct with `TDigest(; compression)`.
- [`push!`] / [`merge!`] — add a value, or fold another digest in.
- [`quantile`] — estimate the value at quantile `q ∈ [0, 1]`.
- [`mean`] / [`var`] / [`std`] — moment estimates over all seen data.
- [`count`] — total weight (number of samples).
- [`serialize`] / [`deserialize`] — compact string form for the
  Redis warm store (cross-shard baseline merge).
"""
module TDigests

import Base: push!, merge!, count, isempty
import Statistics: mean, var, std

export TDigest, quantile, cdf

# A centroid: running mean of a cluster of points + its weight.
mutable struct Centroid
    mean::Float64
    weight::Float64
end

"""
    TDigest(; compression = 100.0, buffer = 256)

Streaming quantile sketch. `compression` (δ) trades memory/accuracy:
higher = more centroids = tighter tails. `buffer` singletons are
accumulated before a batch merge.
"""
mutable struct TDigest
    compression::Float64
    centroids::Vector{Centroid}     # sorted by mean
    buffer::Vector{Float64}         # unmerged singletons
    buffer_cap::Int
    total_weight::Float64
    # Streaming moments (exact, independent of compression) so the
    # zscore path doesn't pay quantile error.
    n::Int
    sum::Float64
    sumsq::Float64
    min::Float64
    max::Float64
end

function TDigest(; compression::Real = 100.0, buffer::Integer = 256)
    return TDigest(Float64(compression), Centroid[], Float64[], Int(buffer),
                   0.0, 0, 0.0, 0.0, Inf, -Inf)
end

isempty(d::TDigest) = d.n == 0
count(d::TDigest)   = d.n

# ---------------------------------------------------------------------------
# Ingestion.
# ---------------------------------------------------------------------------

"""
    push!(d::TDigest, x::Real) -> d

Add one observation. Amortized O(1); triggers a compress when the
singleton buffer fills.
"""
function push!(d::TDigest, x::Real)
    xf = Float64(x)
    Base.push!(d.buffer, xf)
    d.n   += 1
    d.sum += xf
    d.sumsq += xf * xf
    d.min = min(d.min, xf)
    d.max = max(d.max, xf)
    length(d.buffer) >= d.buffer_cap && _flush!(d)
    return d
end

# Fold the buffered singletons into the centroid list and re-compress.
function _flush!(d::TDigest)
    isempty(d.buffer) && return d
    for x in d.buffer
        Base.push!(d.centroids, Centroid(x, 1.0))
        d.total_weight += 1.0
    end
    empty!(d.buffer)
    _compress!(d)
    return d
end

# Merge-compress: sort centroids, then greedily coalesce adjacent ones
# while the running quantile stays inside the k1 size bound.
function _compress!(d::TDigest)
    isempty(d.centroids) && return d
    sort!(d.centroids; by = c -> c.mean)
    total = d.total_weight
    total <= 0 && return d

    merged = Centroid[]
    q0 = 0.0                       # cumulative weight fraction so far
    cur = Centroid(d.centroids[1].mean, d.centroids[1].weight)
    for i in 2:length(d.centroids)
        c = d.centroids[i]
        # Max weight the current centroid may hold at this quantile.
        q_limit = _k_inv(_k(q0, d.compression) + 1.0, d.compression)
        max_w = (q_limit - q0) * total
        if cur.weight + c.weight <= max_w
            # Coalesce: weighted-mean update.
            w = cur.weight + c.weight
            cur.mean = (cur.mean * cur.weight + c.mean * c.weight) / w
            cur.weight = w
        else
            Base.push!(merged, cur)
            q0 += cur.weight / total
            cur = Centroid(c.mean, c.weight)
        end
    end
    Base.push!(merged, cur)
    d.centroids = merged
    return d
end

# k1 scale function and its inverse (bounds centroid sizes; tails small).
_k(q::Float64, δ::Float64) = δ / (2π) * asin(2 * clamp(q, 0.0, 1.0) - 1)
_k_inv(k::Float64, δ::Float64) = (sin(clamp(k / (δ / (2π)), -π/2, π/2)) + 1) / 2

# ---------------------------------------------------------------------------
# Merge (cross-shard baseline).
# ---------------------------------------------------------------------------

"""
    merge!(d::TDigest, other::TDigest) -> d

Fold `other` into `d`. Exact for the moment accumulators; the
centroid lists are concatenated and re-compressed. Used by the Redis
warm store to combine per-shard baselines.
"""
function merge!(d::TDigest, other::TDigest)
    _flush!(other)
    for c in other.centroids
        Base.push!(d.centroids, Centroid(c.mean, c.weight))
        d.total_weight += c.weight
    end
    d.n     += other.n
    d.sum   += other.sum
    d.sumsq += other.sumsq
    d.min = min(d.min, other.min)
    d.max = max(d.max, other.max)
    _compress!(d)
    return d
end

# ---------------------------------------------------------------------------
# Queries.
# ---------------------------------------------------------------------------

"""
    quantile(d::TDigest, q::Real) -> Float64

Estimate the value at quantile `q ∈ [0, 1]`. Returns `NaN` on an
empty digest. Interpolates linearly between centroid means, clamped
to the observed `[min, max]`.
"""
function quantile(d::TDigest, q::Real)
    _flush!(d)
    isempty(d.centroids) && return NaN
    qf = clamp(Float64(q), 0.0, 1.0)
    total = d.total_weight
    length(d.centroids) == 1 && return d.centroids[1].mean

    target = qf * total
    # Walk cumulative weight; centroid i "covers" [cum, cum+w]. Use the
    # centroid-center convention: centroid i's mean sits at cum + w/2.
    cum = 0.0
    for (i, c) in enumerate(d.centroids)
        center = cum + c.weight / 2
        if target <= center
            if i == 1
                # Left tail: interpolate from the min to the first center.
                left_center = d.centroids[1].weight / 2
                left_center <= 0 && return c.mean
                t = target / left_center
                return d.min + t * (c.mean - d.min)
            else
                prev = d.centroids[i - 1]
                prev_center = cum - prev.weight / 2
                span = center - prev_center
                span <= 0 && return c.mean
                t = (target - prev_center) / span
                return prev.mean + t * (c.mean - prev.mean)
            end
        end
        cum += c.weight
    end
    # Right tail.
    last = d.centroids[end]
    last_center = total - last.weight / 2
    span = total - last_center
    span <= 0 && return last.mean
    t = (target - last_center) / span
    return last.mean + t * (d.max - last.mean)
end

"""
    cdf(d::TDigest, x::Real) -> Float64

Estimate P(X ≤ x). Inverse of [`quantile`]; useful for turning a raw
score into a rank/percentile. Returns `NaN` on an empty digest.
"""
function cdf(d::TDigest, x::Real)
    _flush!(d)
    isempty(d.centroids) && return NaN
    xf = Float64(x)
    total = d.total_weight
    xf <= d.min && return 0.0
    xf >= d.max && return 1.0
    cum = 0.0
    for (i, c) in enumerate(d.centroids)
        if xf < c.mean
            center = cum + c.weight / 2
            if i == 1
                left = d.min
                span = c.mean - left
                span <= 0 && return center / total
                t = (xf - left) / span
                return (t * (c.weight / 2)) / total
            else
                prev = d.centroids[i - 1]
                prev_center = cum - prev.weight / 2
                span = c.mean - prev.mean
                span <= 0 && return center / total
                t = (xf - prev.mean) / span
                return (prev_center + t * (center - prev_center)) / total
            end
        end
        cum += c.weight
    end
    return 1.0
end

"Exact running mean over all pushed values."
mean(d::TDigest) = d.n == 0 ? NaN : d.sum / d.n

"Exact running (population) variance."
function var(d::TDigest)
    d.n < 2 && return NaN
    μ = d.sum / d.n
    return max(0.0, d.sumsq / d.n - μ * μ)
end

std(d::TDigest) = sqrt(var(d))

# ---------------------------------------------------------------------------
# Serialization (Redis warm store).
# ---------------------------------------------------------------------------

"""
    serialize(d::TDigest) -> String

Compact, self-describing text form: `compression;n;sum;sumsq;min;max;
mean1:weight1,mean2:weight2,...`. Round-trips through [`deserialize`].
Used to store the digest in a Redis string for cross-shard merging.
"""
function serialize(d::TDigest)
    _flush!(d)
    cents = join(("$(c.mean):$(c.weight)" for c in d.centroids), ",")
    return string(d.compression, ";", d.n, ";", d.sum, ";", d.sumsq, ";",
                  d.min, ";", d.max, ";", cents)
end

"""
    deserialize(s::AbstractString) -> TDigest

Rebuild a digest from [`serialize`] output. Malformed input raises
`ArgumentError`.
"""
function deserialize(s::AbstractString)
    parts = split(String(s), ';')
    length(parts) == 7 ||
        throw(ArgumentError("malformed TDigest string"))
    d = TDigest(; compression = parse(Float64, parts[1]))
    d.n     = parse(Int, parts[2])
    d.sum   = parse(Float64, parts[3])
    d.sumsq = parse(Float64, parts[4])
    d.min   = parse(Float64, parts[5])
    d.max   = parse(Float64, parts[6])
    if !isempty(parts[7])
        for tok in split(parts[7], ',')
            mw = split(tok, ':')
            length(mw) == 2 || continue
            c = Centroid(parse(Float64, mw[1]), parse(Float64, mw[2]))
            Base.push!(d.centroids, c)
            d.total_weight += c.weight
        end
    end
    return d
end

end # module TDigests
