"""
    Anomaly.Instance

Port of the thesis's instance-level anomaly detection (§3.2.5) — score a
single log event by how badly the trained autoencoder reconstructs it.
Works with any Lux `Chain` whose input and output dimensions match; in
practice paired with `DeepKATE`.

Three scalar-per-sample metrics are provided:

- [`reconstruction_error_abs`] — L1, matches Gleichung 3.3 (`E_abs =
  Σ|y − ŷ|`), the thesis's primary reconstruction signal.
- [`reconstruction_error_sq`] — L2-squared; useful when outliers should
  be penalised more than moderate deviations.
- [`latent_distance`] — Euclidean distance between the latent
  embedding and a reference point (typically the cluster centroid or
  the batch mean).

[`anomaly_score`] combines them with user-supplied weights into the
meta-metric the thesis describes ("gewichtet zu einer Meta-Metrik
zusammengeführt").

All functions accept test-mode states so the stochastic layers
(`Dropout`, the KATE competition) are bypassed — this matches the
thesis's workflow of training the AE first, then scoring events with
the frozen network.
"""
module Instance

using Lux
using Statistics
using ..DeepKATE: latent_layer
using ..Masking: SlotValue

export reconstruction_error_abs, reconstruction_error_sq,
       latent_distance, anomaly_score,
       ValueNoveltyDetector, update!, value_novelty, combined_anomaly

# ---------------------------------------------------------------------------
# Reconstruction errors
# ---------------------------------------------------------------------------

@inline function _forward(model, ps, st, X::AbstractArray)
    y, _ = model(X, ps, Lux.testmode(st))
    return y
end

"""
    reconstruction_error_abs(model, ps, st, X) -> Vector

Per-sample absolute reconstruction error `Σⱼ |Xⱼ − ŷⱼ|` (Gleichung 3.3
in the thesis). `X` is an `(n, batch)` matrix; returns a vector of
length `batch`. Uses test-mode state so dropout and KATE competition
are disabled.
"""
function reconstruction_error_abs(model, ps, st, X::AbstractMatrix)
    Y = _forward(model, ps, st, X)
    return vec(sum(abs, X .- Y; dims = 1))
end

reconstruction_error_abs(model, ps, st, x::AbstractVector) =
    only(reconstruction_error_abs(model, ps, st, reshape(x, :, 1)))

"""
    reconstruction_error_sq(model, ps, st, X) -> Vector

Per-sample squared reconstruction error `Σⱼ (Xⱼ − ŷⱼ)²`. Penalises
large deviations more than `reconstruction_error_abs` and is useful
when the score feeds into a thresholded detector.
"""
function reconstruction_error_sq(model, ps, st, X::AbstractMatrix)
    Y = _forward(model, ps, st, X)
    return vec(sum(abs2, X .- Y; dims = 1))
end

reconstruction_error_sq(model, ps, st, x::AbstractVector) =
    only(reconstruction_error_sq(model, ps, st, reshape(x, :, 1)))

# ---------------------------------------------------------------------------
# Latent-space distance
# ---------------------------------------------------------------------------

"""
    latent_distance(model, ps, st, X; reference = :mean, lat = latent_layer(model))
        -> Vector

Per-sample Euclidean distance in the encoder's latent space. `reference`
is either `:mean` (use the batch's latent centroid — useful for
"distance to the crowd" anomaly scoring) or a vector/matrix of latent
reference points.
"""
function latent_distance(
    model, ps, st, X::AbstractMatrix;
    reference = :mean,
    lat::Int = latent_layer(model),
)
    embeddings = _encode(model, ps, st, X, lat)
    ref = reference === :mean ? vec(mean(embeddings; dims = 2)) : reference
    if ref isa AbstractVector
        return vec(sqrt.(sum(abs2, embeddings .- ref; dims = 1)))
    else
        ref::AbstractMatrix
        size(ref) == size(embeddings) ||
            throw(DimensionMismatch("reference matrix must be (latent, batch)"))
        return vec(sqrt.(sum(abs2, embeddings .- ref; dims = 1)))
    end
end

latent_distance(model, ps, st, x::AbstractVector; kwargs...) =
    only(latent_distance(model, ps, st, reshape(x, :, 1); kwargs...))

function _encode(model, ps, st, X::AbstractMatrix, lat::Int)
    st_test = Lux.testmode(st)
    out = X
    for i in 1:lat
        sym = Symbol(:layer_, i)
        layer = getfield(model.layers, sym)
        p = getfield(ps, sym)
        s = getfield(st_test, sym)
        out, _ = layer(out, p, s)
    end
    return out
end

# ---------------------------------------------------------------------------
# Meta-metric
# ---------------------------------------------------------------------------

"""
    anomaly_score(model, ps, st, X;
                  weights = (abs = 0.5, sq = 0.5, latent = 0.0),
                  reference = :mean,
                  normalise = true) -> Vector

Weighted combination of the three metrics — the thesis's "Meta-Metrik".
Components with weight `0` are skipped (and their forward pass is
avoided). When `normalise = true` each component is divided by its own
batch-max before the weighted sum, so the score is in `[0, 1]` and
invariant to absolute magnitudes.

Weights are supplied as a NamedTuple for readability; the default
averages `reconstruction_error_abs` and `reconstruction_error_sq` and
ignores latent distance.
"""
function anomaly_score(
    model, ps, st, X::AbstractMatrix;
    weights = (abs = 0.5, sq = 0.5, latent = 0.0),
    reference = :mean,
    normalise::Bool = true,
)
    batch = size(X, 2)
    score = zeros(Float64, batch)
    total_w = 0.0

    if weights.abs != 0
        s = reconstruction_error_abs(model, ps, st, X)
        score .+= weights.abs .* _maybe_normalise(s, normalise)
        total_w += weights.abs
    end
    if weights.sq != 0
        s = reconstruction_error_sq(model, ps, st, X)
        score .+= weights.sq .* _maybe_normalise(s, normalise)
        total_w += weights.sq
    end
    if weights.latent != 0
        s = latent_distance(model, ps, st, X; reference = reference)
        score .+= weights.latent .* _maybe_normalise(s, normalise)
        total_w += weights.latent
    end
    total_w > 0 || throw(ArgumentError("anomaly_score: all weights are zero"))
    return score ./ total_w
end

anomaly_score(model, ps, st, x::AbstractVector; kwargs...) =
    only(anomaly_score(model, ps, st, reshape(x, :, 1); kwargs...))

@inline function _maybe_normalise(s::AbstractVector, on::Bool)
    on || return Float64.(s)
    m = maximum(s)
    return m > 0 ? Float64.(s) ./ Float64(m) : Float64.(s)
end

# ---------------------------------------------------------------------------
# Value-novelty — catch outliers that the template route hides
# ---------------------------------------------------------------------------
#
# The masking pipeline replaces every typed slot with a `<LABEL>`
# placeholder before clustering, which means a never-before-seen IP or
# a wildly out-of-range number looks exactly like every other IP / NUM
# to the AE. This detector runs *in parallel* to the reconstruction
# score and flags lines whose slot *values* are new or anomalous
# within their label's distribution.
#
# Two signals:
# - Novelty (categorical): fraction of slot values never observed for
#   that label before. Captures new IPs, new user names, new paths.
# - Range (numeric): absolute z-score of the slot value under the
#   running mean/variance of its label's empirical distribution.
#   Captures out-of-range numbers / durations / sizes.
#
# Stateful on purpose — calling code decides when to `update!` (e.g.
# during a training-window pass) and when to score without updating.

"""
    ValueNoveltyDetector()

Per-label memory of seen values + running mean/variance for numeric
labels. Call [`update!`] on each training-window line, then
[`value_novelty`] to score new lines; [`combined_anomaly`] fuses the
template-reconstruction score with the value-novelty signal.
"""
mutable struct ValueNoveltyDetector
    seen::Dict{String, Set{String}}
    counts::Dict{String, Int}
    sum::Dict{String, Float64}
    sumsq::Dict{String, Float64}
    n_numeric::Dict{String, Int}
    ValueNoveltyDetector() = new(
        Dict{String, Set{String}}(),
        Dict{String, Int}(),
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        Dict{String, Int}(),
    )
end

@inline function _parse_float(s::AbstractString)
    # Accept both EN `3.14` and DE `3,14` fractional styles. Reject
    # grouping-only integers that happen to contain separators.
    s2 = replace(s, ',' => '.')
    try
        return parse(Float64, s2)
    catch
        return nothing
    end
end

"""
    update!(det::ValueNoveltyDetector, values::Vector{SlotValue})

Fold a line's slot values into the detector's per-label memory. Safe
to call many times; the detector grows monotonically.
"""
function update!(det::ValueNoveltyDetector, values::AbstractVector{SlotValue})
    @inbounds for v in values
        push!(get!(det.seen, v.label, Set{String}()), v.value)
        det.counts[v.label] = get(det.counts, v.label, 0) + 1
        f = _parse_float(v.value)
        if f !== nothing
            det.sum[v.label] = get(det.sum, v.label, 0.0) + f
            det.sumsq[v.label] = get(det.sumsq, v.label, 0.0) + f * f
            det.n_numeric[v.label] = get(det.n_numeric, v.label, 0) + 1
        end
    end
    return det
end

"""
    value_novelty(det, values; numeric_sigma = 3.0) -> Float64

Score one line's slot values under the detector. Returns a non-negative
float in `[0, 1+]`:

- Each value not previously seen for its label contributes `1 /
  |values|`.
- Each numeric value with running-mean z-score `|z| > numeric_sigma`
  contributes an additional `min(|z| / numeric_sigma, 1) /
  |values|`.

`0.0` means every slot is familiar; `≥ 1.0` means every slot is
either unseen or strongly out-of-range. Call before [`update!`] on
test-window lines to keep evaluation honest.
"""
function value_novelty(det::ValueNoveltyDetector,
                       values::AbstractVector{SlotValue};
                       numeric_sigma::Real = 3.0)
    isempty(values) && return 0.0
    n = length(values)
    total = 0.0
    @inbounds for v in values
        seen_set = get(det.seen, v.label, nothing)
        unseen = seen_set === nothing || !(v.value in seen_set)
        if unseen
            total += 1.0
        end
        f = _parse_float(v.value)
        if f !== nothing
            k = get(det.n_numeric, v.label, 0)
            if k >= 2
                μ = det.sum[v.label] / k
                σ² = max(0.0, det.sumsq[v.label] / k - μ^2)
                σ = sqrt(σ²)
                if σ > 0
                    z = abs(f - μ) / σ
                    if z > numeric_sigma
                        total += min(z / numeric_sigma, 1.0)
                    end
                end
            end
        end
    end
    return total / n
end

"""
    combined_anomaly(model, ps, st, X, per_line_values, det;
                     weights = (template = 0.5, values = 0.5),
                     score_kwargs...) -> Vector{Float64}

Weighted combination of the template-reconstruction score
([`anomaly_score`]) and the [`value_novelty`] signal. `X` is the
templated-feature matrix (what the AE sees), `per_line_values` is
the vector-of-vectors of [`SlotValue`]s from
[`LogClustering.Masking.mask_lines_with_values`], `det` is a
[`ValueNoveltyDetector`] trained on a clean window. Returns one score
per sample; higher is more anomalous.

The weights are renormalised internally so `weights.template = 0`
gives pure value-outlier scoring and `weights.values = 0` recovers
pure template-reconstruction scoring.
"""
function combined_anomaly(model, ps, st, X::AbstractMatrix,
                          per_line_values::AbstractVector{<:AbstractVector{SlotValue}},
                          det::ValueNoveltyDetector;
                          weights = (template = 0.5, values = 0.5),
                          score_kwargs...)
    size(X, 2) == length(per_line_values) ||
        throw(DimensionMismatch("X columns ($(size(X, 2))) != values rows ($(length(per_line_values)))"))
    template_score = anomaly_score(model, ps, st, X; score_kwargs...)
    value_score = Float64[value_novelty(det, per_line_values[j])
                          for j in 1:size(X, 2)]
    total_w = weights.template + weights.values
    total_w > 0 || throw(ArgumentError("both weights are zero"))
    return (weights.template .* Float64.(template_score) .+
            weights.values   .* value_score) ./ total_w
end

end # module Instance
