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

export reconstruction_error_abs, reconstruction_error_sq,
       latent_distance, anomaly_score

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

end # module Instance
