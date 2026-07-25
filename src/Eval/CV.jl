"""
    Eval.CV

Cross-validation splits for plan 001 Stage F. The plan explicitly
forbids random k-fold on log data because log lines from the same
application / host have strong temporal and syntactic leakage —
shuffling them lets the model memorise the split's identifiers.

This module provides:

- [`Split`] — a `(train, test)` pair of 1-based index vectors.
- [`time_ordered_split`] — one trailing-window split (holdout).
- [`time_ordered_kfold`] — walk-forward k-fold: each fold trains on
  everything before it and tests on the fold itself.
- [`per_host_split`] — leave one (or more) host(s) out.
- [`stratified_by`] — stratified random split that preserves label
  proportions, for use on the already-label-balanced LogHub-2.0
  2k subsets where time ordering isn't meaningful.

Nothing here runs a model or a parser — the splits are *index sets*
callers apply to their own data vectors.
"""
module CV

using Random

export Split, time_ordered_split, time_ordered_kfold,
       per_host_split, stratified_by

# ---------------------------------------------------------------------------
# Core split type
# ---------------------------------------------------------------------------

"""
    Split(train, test)

A pair of *disjoint* 1-based index vectors into some host vector. The
invariant `isempty(intersect(train, test))` is enforced by every
constructor here.
"""
struct Split
    train::Vector{Int}
    test::Vector{Int}
    function Split(train::AbstractVector{<:Integer}, test::AbstractVector{<:Integer})
        tr = Int[Int(i) for i in train]
        te = Int[Int(i) for i in test]
        isempty(intersect(tr, te)) ||
            throw(ArgumentError("train and test index sets overlap"))
        return new(tr, te)
    end
end

Base.length(s::Split) = length(s.train) + length(s.test)

function Base.show(io::IO, s::Split)
    print(io, "Split(n_train=", length(s.train),
          ", n_test=", length(s.test), ")")
end

# ---------------------------------------------------------------------------
# Time-ordered holdout
# ---------------------------------------------------------------------------

"""
    time_ordered_split(n::Integer; train_frac = 0.8) -> Split

Use the first `train_frac` of `1:n` as the training set and the
remainder as the test set. Preserves temporal ordering — the test
set is strictly *after* the training set.
"""
function time_ordered_split(n::Integer; train_frac::Real = 0.8)
    n >= 2 || throw(ArgumentError("n must be ≥ 2; got $n"))
    0 < train_frac < 1 || throw(ArgumentError("train_frac must be in (0, 1)"))
    cut = max(1, min(n - 1, floor(Int, train_frac * n)))
    return Split(1:cut, cut+1:n)
end

# ---------------------------------------------------------------------------
# Walk-forward k-fold
# ---------------------------------------------------------------------------

"""
    time_ordered_kfold(n::Integer, k::Integer; min_train = 1) -> Vector{Split}

Walk-forward k-fold. Splits `1:n` into `k` roughly equal contiguous
folds; fold `i`'s test set is fold `i`, and its train set is everything
in folds `1..i-1`. Fold 1 has empty train (skipped unless
`min_train = 0`), so the returned vector has `k-1` splits by default.

Useful for log-parser evaluation where performance should grow
monotonically with the training prefix.
"""
function time_ordered_kfold(n::Integer, k::Integer; min_train::Integer = 1)
    k >= 2 || throw(ArgumentError("k must be ≥ 2"))
    n >= k || throw(ArgumentError("need n ≥ k; got n=$n, k=$k"))
    # Fold boundaries: fold i = boundaries[i]+1 : boundaries[i+1].
    boundaries = Int[round(Int, j * n / k) for j in 0:k]
    splits = Split[]
    for i in 2:k
        train = (boundaries[1] + 1):boundaries[i]
        test  = (boundaries[i] + 1):boundaries[i + 1]
        if length(train) < min_train || isempty(test)
            continue
        end
        push!(splits, Split(train, test))
    end
    return splits
end

# ---------------------------------------------------------------------------
# Per-host leave-one-(group)-out
# ---------------------------------------------------------------------------

"""
    per_host_split(hosts; held_out) -> Split

Leave-one-host-out (or leave-k-hosts-out) split. `hosts` is a vector
of host identifiers, one per log line. `held_out` is either a single
host id or a collection; every line whose host is in `held_out` lands
in the test set, the rest in train.
"""
function per_host_split(hosts::AbstractVector{T}; held_out) where {T}
    held = held_out isa AbstractVector || held_out isa Set ?
        Set(convert(Vector{T}, collect(held_out))) :
        Set([convert(T, held_out)])
    train = Int[]
    test  = Int[]
    @inbounds for (i, h) in enumerate(hosts)
        if h in held
            push!(test, i)
        else
            push!(train, i)
        end
    end
    isempty(test) && throw(ArgumentError(
        "no lines matched the held-out hosts; nothing to test on"))
    isempty(train) && throw(ArgumentError(
        "every line is held out; nothing to train on"))
    return Split(train, test)
end

"""
    per_host_kfold(hosts; rng = default_rng(), k = 5) -> Vector{Split}

Partition the *unique* hosts into `k` groups and emit one leave-one-group-out
split per group. Unlike `time_ordered_kfold`, each fold's train set
mixes hosts in time — the only guaranteed separation is by host
identity, which is the plan's explicit non-leakage requirement.
"""
function per_host_kfold(hosts::AbstractVector{T}; rng::AbstractRNG = Random.default_rng(),
                        k::Integer = 5) where {T}
    k >= 2 || throw(ArgumentError("k must be ≥ 2"))
    uniq = collect(Set(hosts))
    length(uniq) >= k || throw(ArgumentError("need at least k distinct hosts; \
                                              got $(length(uniq)) for k=$k"))
    shuffled = uniq[randperm(rng, length(uniq))]
    groups = [T[] for _ in 1:k]
    for (i, h) in enumerate(shuffled)
        push!(groups[mod1(i, k)], h)
    end
    return [per_host_split(hosts; held_out = g) for g in groups]
end

export per_host_kfold

# ---------------------------------------------------------------------------
# Stratified random split
# ---------------------------------------------------------------------------

"""
    stratified_by(labels; train_frac = 0.8, rng = default_rng()) -> Split

Random train/test split that preserves the frequency of every label
within ±1 sample. Use this on *already curated* benchmarks
(e.g. LogHub-2.0 2k subsets) where time order isn't informative and
every template class should appear in both splits.
"""
function stratified_by(labels::AbstractVector;
                       train_frac::Real = 0.8,
                       rng::AbstractRNG = Random.default_rng())
    0 < train_frac < 1 || throw(ArgumentError("train_frac must be in (0, 1)"))
    by_label = Dict{Any, Vector{Int}}()
    @inbounds for (i, l) in enumerate(labels)
        push!(get!(by_label, l, Int[]), i)
    end
    train = Int[]
    test  = Int[]
    for (_, idxs) in by_label
        shuf = idxs[randperm(rng, length(idxs))]
        cut = max(0, min(length(shuf), round(Int, train_frac * length(shuf))))
        append!(train, shuf[1:cut])
        append!(test,  shuf[cut+1:end])
    end
    sort!(train); sort!(test)
    return Split(train, test)
end

end # module CV
