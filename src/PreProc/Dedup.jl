"""
    PreProc.Dedup

Streaming content-hash deduplication — plan 001 Stage E′ step 5.

90 %+ of production log volume is exact duplicates; dedup before
clustering / embedding is free throughput. This module ships a pure
Julia implementation:

- [`DedupState`] — Bloom filter sized for a caller-supplied expected
  count `n` and false-positive rate `fpr`. Parameters `m` (bits) and
  `k` (hash functions) are derived via the standard formulas.
- [`is_new!`] — query + insert in one call; returns `true` the first
  time (according to the filter) a content hash is seen, `false`
  afterwards. False positives (claiming "already seen" for genuinely
  new lines) are bounded by `fpr`; **no** false negatives.
- [`dedup`] — convenience that returns the list of first-seen lines
  in order and a `Vector{Bool}` marking which originals survived.

Uses `Base.hash` with `k` distinct salts, which is Julia stdlib and
cryptographically weak but uniform enough at the filter scales we
care about (< 10⁷ items). Swap to `xxHash_jll` if you need to
handle adversarial / cryptographic duplicates.
"""
module Dedup

export DedupState, is_new!, contains, dedup, fpr_estimate

# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------

"""
Streaming Bloom-filter state. Construct with
`DedupState(; expected_n, fpr = 1e-4)`.

Internal fields:
- `m`    — bit count
- `k`    — number of hash functions
- `bits` — backing `BitVector`
- `observed` — how many insertions have been issued
"""
mutable struct DedupState
    m::Int
    k::Int
    bits::BitVector
    observed::Int
end

function DedupState(; expected_n::Union{Nothing, Integer} = nothing,
                    fpr::Real = 1e-4,
                    m::Union{Nothing, Integer} = nothing,
                    k::Union{Nothing, Integer} = nothing,
                    bits::Union{Nothing, BitVector} = nothing,
                    observed::Integer = 0)
    if m !== nothing && k !== nothing
        # Restore-from-fields path (used by Persistence rehydration).
        bv = bits === nothing ? falses(Int(m)) : bits
        length(bv) == Int(m) ||
            throw(ArgumentError("length(bits) ($(length(bv))) != m ($m)"))
        return DedupState(Int(m), Int(k), bv, Int(observed))
    end
    # Size-from-workload path (the common call).
    expected_n === nothing &&
        throw(ArgumentError("supply either `expected_n` or both `m` and `k`"))
    expected_n >= 1 || throw(ArgumentError("expected_n must be ≥ 1"))
    0 < fpr < 1 || throw(ArgumentError("fpr must be in (0, 1)"))
    # Optimal-size formulas (Mitzenmacher & Upfal 2005).
    m′ = max(8, ceil(Int, -expected_n * log(fpr) / (log(2)^2)))
    k′ = max(1, round(Int, m′ / expected_n * log(2)))
    return DedupState(m′, k′, falses(m′), 0)
end

# Double hashing (Kirsch & Mitzenmacher 2006): two independent Julia
# hashes combine to yield k well-mixed indices. Naïvely salting
# `hash(UInt(i), h)` with small `i` biases the low bits so `k` derived
# indices land on too few buckets — we measured a 5× FPR inflation
# compared to the `(1 - e^{-kn/m})^k` target with that scheme.
const _BLOOM_SALT_1 = 0x9ae16a3b2f90404f
const _BLOOM_SALT_2 = 0xc3a5c85c97cb3127

@inline function _pair(x)
    h1 = hash(x, _BLOOM_SALT_1 % UInt)
    h2 = hash(x, _BLOOM_SALT_2 % UInt)
    # Ensure h2 is odd so the stride sweeps all positions uniformly.
    return h1, h2 | one(UInt)
end

@inline function _index(s::DedupState, i::Int, h1::UInt, h2::UInt)
    return Int((h1 + UInt(i) * h2) % UInt(s.m)) + 1
end

"""
    is_new!(s::DedupState, x) -> Bool

Hash `x` with all `s.k` double-hashed functions; if any corresponding
bit is `0`, flip every bit to `1` and return `true` (first time seen).
If every bit is already `1`, return `false` (probably seen — may be a
false positive bounded by the configured `fpr`).
"""
function is_new!(s::DedupState, x)
    new = false
    h1, h2 = _pair(x)
    @inbounds for i in 1:s.k
        idx = _index(s, i, h1, h2)
        if !s.bits[idx]
            s.bits[idx] = true
            new = true
        end
    end
    s.observed += 1
    return new
end

"""
    contains(s::DedupState, x) -> Bool

Query-only membership test. Returns `true` iff every hashed bit is
already set (i.e. the filter says "probably seen"). Does not modify
state; does not count as an observation.
"""
function contains(s::DedupState, x)
    h1, h2 = _pair(x)
    @inbounds for i in 1:s.k
        if !s.bits[_index(s, i, h1, h2)]
            return false
        end
    end
    return true
end

"""
    fpr_estimate(s::DedupState) -> Float64

Instantaneous estimate of the Bloom filter's false-positive rate
given the fill so far: `(1 - exp(-k·n/m))^k`.
"""
function fpr_estimate(s::DedupState)
    s.observed == 0 && return 0.0
    return (1 - exp(-s.k * s.observed / s.m))^s.k
end

# ---------------------------------------------------------------------------
# Convenience batch API
# ---------------------------------------------------------------------------

"""
    dedup(lines; fpr = 1e-4) -> (unique_lines::Vector, is_first_seen::BitVector)

Streaming dedup over `lines`, returning the first-seen lines in
input order and a per-line mask flagging which originals were new.
Allocates a Bloom filter sized for `length(lines)` at the configured
`fpr`.
"""
function dedup(lines::AbstractVector; fpr::Real = 1e-4)
    n = max(1, length(lines))
    s = DedupState(; expected_n = n, fpr = fpr)
    mask = falses(length(lines))
    uniq = eltype(lines)[]
    @inbounds for (i, l) in enumerate(lines)
        if is_new!(s, l)
            mask[i] = true
            push!(uniq, l)
        end
    end
    return uniq, mask
end

# ---------------------------------------------------------------------------
# Recent-window LRU memo — near-duplicate *result reuse* for the stream.
# ---------------------------------------------------------------------------

"""
    LRUMemo{K, V}(capacity)

Fixed-capacity recent-window cache with FIFO eviction. Used by the
streaming worker to memoize the expensive per-line model output (e.g.
the transformer NLL) keyed by the line's masked template: two lines
with the same typed-slot template mask to the same key and produce an
*identical* masked token sequence, so the cached result is exact —
this collapses the dominant near-duplicate cost with zero risk of
merging genuinely-different lines (unlike SimHash). `hits` / `misses`
counters feed the status heartbeat.
"""
mutable struct LRUMemo{K, V}
    cap::Int
    store::Dict{K, V}
    order::Vector{K}     # front = oldest inserted
    hits::Int
    misses::Int
end

LRUMemo{K, V}(cap::Integer) where {K, V} =
    LRUMemo{K, V}(Int(cap), Dict{K, V}(), K[], 0, 0)

Base.haskey(m::LRUMemo, k) = haskey(m.store, k)
Base.length(m::LRUMemo)    = length(m.store)

"""
    memoize!(m::LRUMemo, key, compute) -> value

Return the cached value for `key`, or call `compute()` (a
zero-argument closure), cache, and return it. FIFO-evicts the oldest
entry past `cap`. Bumps `hits`/`misses`.
"""
function memoize!(m::LRUMemo{K, V}, key, compute) where {K, V}
    kk = convert(K, key)
    v = get(m.store, kk, nothing)
    if v !== nothing || haskey(m.store, kk)
        m.hits += 1
        return m.store[kk]
    end
    m.misses += 1
    val = convert(V, compute())
    m.store[kk] = val
    push!(m.order, kk)
    if length(m.order) > m.cap
        old = popfirst!(m.order)
        delete!(m.store, old)
    end
    return val
end

export LRUMemo, memoize!

end # module Dedup
