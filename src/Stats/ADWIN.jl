"""
    Stats.ADWINs

Pure-Julia ADWIN (ADaptive WINdowing, Bifet & Gavaldà 2007) — an
online change-point detector over a stream of reals. It keeps a
variable-length window of recent values as exponential-histogram
buckets (O(log n) memory / update); whenever the window can be split
into an older and a newer sub-window whose means differ by more than
a Hoeffding-bound `ε_cut`, the older sub-window is dropped and a
change is flagged. The retained window therefore always reflects the
*current* regime, and its mean is an adaptive baseline.

Used by the rule engine's `rate_spike` / `volume_anomaly` rules in
`"baseline": "auto:changepoint"` mode: instead of firing on a fixed
multiple of a frozen baseline, they fire when ADWIN detects an
**upward** regime shift in the windowed count — catching bursts a
static threshold misses (and not crying wolf when the baseline
itself drifts up slowly).

## Surface

- [`ADWIN`] — the detector. `ADWIN(; delta)`.
- [`update!`] — add a value; returns `true` when a change was
  detected (the older window was dropped).
- [`width`] / [`mean`] — current window size / adaptive mean.
- [`last_drop_mean`] — mean of the sub-window most recently dropped
  (so callers can tell an *upward* shift from a downward one).
"""
module ADWINs

import Base: length
import Statistics: mean

export ADWIN, update!, width, last_drop_mean

# One exponential-histogram bucket: `count` items (a power of two)
# with running `total`.
mutable struct Bucket
    count::Int
    total::Float64
end

"""
    ADWIN(; delta = 0.002, max_buckets = 5)

Change detector. `delta` is the confidence parameter (smaller =
fewer false alarms, slower to react). `max_buckets` bounds how many
buckets of each size are kept before merging.
"""
mutable struct ADWIN
    delta::Float64
    max_buckets::Int
    buckets::Vector{Bucket}     # oldest first, counts are powers of two
    total::Float64
    n::Int
    last_drop_mean::Float64
end

function ADWIN(; delta::Real = 0.002, max_buckets::Integer = 5)
    0 < delta < 1 || throw(ArgumentError("delta must be in (0,1)"))
    return ADWIN(Float64(delta), Int(max_buckets), Bucket[], 0.0, 0, NaN)
end

length(a::ADWIN) = a.n
width(a::ADWIN)  = a.n
mean(a::ADWIN)   = a.n == 0 ? NaN : a.total / a.n
last_drop_mean(a::ADWIN) = a.last_drop_mean

"""
    update!(a::ADWIN, x::Real) -> Bool

Insert `x` and test for a distribution change. Returns `true` when a
change was detected (and the stale older window dropped). Amortized
O(log n).
"""
function update!(a::ADWIN, x::Real)
    xf = Float64(x)
    # New size-1 bucket at the newest end.
    push!(a.buckets, Bucket(1, xf))
    a.total += xf
    a.n += 1
    _compress!(a)
    return _detect_change!(a)
end

# Merge to keep at most `max_buckets` buckets of each size (count).
# Two oldest same-count buckets combine into one double-count bucket.
function _compress!(a::ADWIN)
    # Walk from newest to oldest counting runs of equal `count`.
    # Simpler correct approach: repeatedly find the oldest pair of
    # adjacent equal-count buckets once a size has > max_buckets.
    changed = true
    while changed
        changed = false
        # Count buckets by size.
        i = 1
        n = length(a.buckets)
        while i <= n
            c = a.buckets[i].count
            # run of equal counts is contiguous (invariant: sizes are
            # non-increasing from oldest→? actually non-decreasing newest
            # end). Count how many buckets share this count.
            j = i
            while j <= n && a.buckets[j].count == c
                j += 1
            end
            run = j - i
            if run > a.max_buckets
                # Merge the two OLDEST of this run (positions i, i+1).
                b1 = a.buckets[i]; b2 = a.buckets[i + 1]
                merged = Bucket(b1.count + b2.count, b1.total + b2.total)
                deleteat!(a.buckets, i + 1)
                a.buckets[i] = merged
                changed = true
                break
            end
            i = j
        end
    end
    return nothing
end

# Try to shrink the window: for each split boundary (oldest side),
# test the Hoeffding cut; on a hit drop the oldest bucket and flag.
function _detect_change!(a::ADWIN)
    detected = false
    shrunk = true
    while shrunk && length(a.buckets) >= 2
        shrunk = false
        n0 = 0; sum0 = 0.0
        # Accumulate the older sub-window bucket by bucket.
        for k in 1:(length(a.buckets) - 1)
            n0 += a.buckets[k].count
            sum0 += a.buckets[k].total
            n1 = a.n - n0
            sum1 = a.total - sum0
            (n0 == 0 || n1 == 0) && continue
            m0 = sum0 / n0
            m1 = sum1 / n1
            if abs(m0 - m1) > _epsilon_cut(a, n0, n1)
                # Drop the oldest bucket; record what we dropped.
                dropped = a.buckets[1]
                a.last_drop_mean = dropped.total / dropped.count
                a.total -= dropped.total
                a.n    -= dropped.count
                deleteat!(a.buckets, 1)
                detected = true
                shrunk = true
                break
            end
        end
    end
    return detected
end

# Hoeffding bound with Bonferroni correction over the window
# (the "ADWIN" cut). `m` is the harmonic mean of the two sub-window
# sizes.
function _epsilon_cut(a::ADWIN, n0::Int, n1::Int)
    m = 1.0 / (1.0 / n0 + 1.0 / n1)
    δ′ = a.delta / max(1, a.n)          # per-window Bonferroni
    return sqrt((1.0 / (2.0 * m)) * log(4.0 / δ′))
end

end # module ADWINs
