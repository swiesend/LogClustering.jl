"""
    LogClustering.RCA

End-to-end Root-Cause-Analysis (RCA) pipeline on top of the
existing Stage C / E / E″ pieces. Given a trained embedder
(`DeepKATE` / `VQ-VAE`) and an optional `ValueNoveltyDetector`,
`root_cause(model_bundle, detector_bundle, lines)` returns an
[`RCAReport`] holding:

1. **Cluster ids** — every line's cluster assignment, produced by
   running the saved encoder on the masked BoW features and then
   running k-means on the L2-normalised latent.
2. **Per-line anomaly score** — reconstruction L1 error fused
   with value-novelty (via `Anomaly.Instance.combined_anomaly`)
   when a detector is supplied; pure reconstruction otherwise.
3. **Episodes** — serial patterns mined from the cluster-id
   sequence. Seeded with the cluster ids that appear in the
   top-decile-most-anomalous lines so the miner explicitly looks
   for recurring context around anomalies.
4. **Ranked root causes** — each episode ordered by
   `support × mean_anomaly_density` (density = mean anomaly score
   of the lines the episode's occurrences touch). The top-ranked
   patterns are the explanation: "whenever clusters X, Y, Z fire
   together within gap ≤ max_gap, lines in this window show up
   as anomalous."

Composes existing surface — no kernel changes. The RCA module
is pure orchestration.
"""
module RCA

using Statistics: mean
using DataStructures: OrderedDict
using Lux
using ..Masking: mask_lines_with_values
using ..Featurise: bow
using ..DeepKATE: latent_layer
using ..Pipeline: l2_normalise, kmeans_cluster
using ..Instance: Instance, ValueNoveltyDetector, anomaly_score,
                  combined_anomaly, reconstruction_error_abs
using ..Episodes: mv_span

export RCAReport, root_cause, render_markdown

"""
    RCAReport

Result of a single RCA run. Fields:

- `cluster_ids::Vector{Int}` — per-line cluster assignment (`1:k`).
- `per_line_score::Vector{Float64}` — anomaly score per line.
- `threshold::Float64` — the score cut-off separating "anomalous"
  from "normal" (quantile `top_percentile`).
- `episodes::OrderedDict{Vector{Int}, Vector{Vector{Int}}}` — raw
  output of `Episodes.mv_span` seeded with the top anomaly
  cluster ids.
- `per_episode_density::Dict{Vector{Int}, Float64}` — mean
  per-line anomaly score across the lines each episode's
  occurrences touch.
- `ranked::Vector{@NamedTuple{pattern::Vector{Int}, support::Int,
  density::Float64, score::Float64}}` — episodes sorted by
  `support · density`, highest first.
- `metadata::Dict{String, Any}` — `n_lines`, `n_clusters`,
  `top_percentile`, `min_sup`, `max_gap`, … for provenance.
"""
struct RCAReport
    cluster_ids::Vector{Int}
    per_line_score::Vector{Float64}
    threshold::Float64
    episodes::OrderedDict{Vector{Int}, Vector{Vector{Int}}}
    per_episode_density::Dict{Vector{Int}, Float64}
    ranked::Vector{@NamedTuple{pattern::Vector{Int}, support::Int,
                               density::Float64, score::Float64}}
    metadata::Dict{String, Any}
end

"""
    root_cause(model, ps, st, vocab, lines;
               detector = nothing,
               k_clusters = nothing,
               top_percentile = 0.10,
               min_sup = 3, max_gap = 20,
               max_time_duration = 50,
               rng = Random.default_rng()) -> RCAReport

Run the full pipeline: mask → BoW → encode → cluster → score
→ mine episodes → rank. `model`, `ps`, `st`, and `vocab` come
from a rehydrated DeepKATE bundle; `detector` is an optional
`ValueNoveltyDetector` from a separate bundle. `lines` is the
raw log corpus (vector of strings).

`k_clusters` defaults to `max(4, ceil(sqrt(length(lines))))` — a
standard rule of thumb. `top_percentile` controls the anomaly
cut-off; `min_sup` + `max_gap` + `max_time_duration` flow into
`Episodes.mv_span`.
"""
function root_cause(model, ps, st, vocab,
                    lines::AbstractVector{<:AbstractString};
                    detector::Union{Nothing, ValueNoveltyDetector} = nothing,
                    k_clusters::Union{Nothing, Integer} = nothing,
                    top_percentile::Real = 0.10,
                    min_sup::Integer = 3,
                    max_gap::Integer = 20,
                    max_time_duration::Integer = 50)
    isempty(lines) && throw(ArgumentError("root_cause needs at least one line"))
    (0 < top_percentile < 1) ||
        throw(ArgumentError("top_percentile must be in (0, 1); got $top_percentile"))

    # --- Mask + featurise (using the bundle's vocab) ---------------------
    templates, per_line_values = mask_lines_with_values(lines)
    X = bow(templates, vocab; normalise = :l1)

    # --- Encode through the first `latent_layer(model)` layers ----------
    st_eval = Lux.testmode(st)
    Z = X
    for i in 1:latent_layer(model)
        sym = Symbol(:layer_, i)
        Z, _ = getfield(model.layers, sym)(Z, getfield(ps, sym),
                                           getfield(st_eval, sym))
    end
    Zn = l2_normalise(Z)

    # --- k-means clustering on the L2-normalised latent -----------------
    n_lines = length(lines)
    k = k_clusters === nothing ?
        max(4, ceil(Int, sqrt(n_lines))) : Int(k_clusters)
    k = min(k, n_lines)
    assignments, _, _ = kmeans_cluster(Zn, k)
    cluster_ids = Vector{Int}(assignments)

    # --- Per-line anomaly score -----------------------------------------
    score = if detector === nothing
        Float64.(reconstruction_error_abs(model, ps, st, X))
    else
        Float64.(combined_anomaly(model, ps, st, X, per_line_values,
                                  detector;
                                  weights = (template = 0.5, values = 0.5)))
    end

    # --- Threshold at the requested percentile --------------------------
    sorted = sort(score)
    cutoff_idx = max(1, ceil(Int, (1 - top_percentile) * length(sorted)))
    threshold = sorted[cutoff_idx]
    anom_mask = score .>= threshold
    anom_cids = unique(cluster_ids[anom_mask])

    # --- Episode mining seeded with anomalous cluster ids ---------------
    seeds = [[cid] for cid in anom_cids]
    episodes = if isempty(seeds)
        OrderedDict{Vector{Int}, Vector{Vector{Int}}}()
    else
        mv_span(cluster_ids;
                prefixes = seeds,
                min_sup = min_sup,
                max_gap = max_gap,
                max_time_duration = max_time_duration)
    end

    # --- Per-episode anomaly density -----------------------------------
    per_ep_density = Dict{Vector{Int}, Float64}()
    for (pattern, occurrences) in episodes
        touched = Int[]
        for occ in occurrences
            append!(touched, occ)
        end
        unique!(touched)
        per_ep_density[pattern] = isempty(touched) ? 0.0 :
                                  mean(@view score[touched])
    end

    # --- Rank: support · density, descending ---------------------------
    ranked = @NamedTuple{pattern::Vector{Int}, support::Int,
                        density::Float64, score::Float64}[]
    for (pattern, occurrences) in episodes
        support = length(occurrences)
        density = per_ep_density[pattern]
        push!(ranked, (pattern = pattern, support = support,
                       density = density, score = support * density))
    end
    sort!(ranked; by = r -> r.score, rev = true)

    metadata = Dict{String, Any}(
        "n_lines" => n_lines,
        "n_clusters" => k,
        "top_percentile" => top_percentile,
        "threshold" => threshold,
        "n_anomalies" => count(anom_mask),
        "min_sup" => min_sup,
        "max_gap" => max_gap,
        "max_time_duration" => max_time_duration,
        "with_detector" => detector !== nothing,
    )

    return RCAReport(cluster_ids, score, threshold, episodes,
                     per_ep_density, ranked, metadata)
end

"""
    render_markdown(rep::RCAReport; topk = 10,
                    lines::Union{Nothing, AbstractVector{<:AbstractString}} = nothing)
        -> String

Human-readable Markdown summary. If `lines` is supplied, the top
`topk` episodes each get a "representative lines" snippet (the
first line each occurrence lands on).
"""
function render_markdown(rep::RCAReport; topk::Integer = 10,
                         lines::Union{Nothing, AbstractVector{<:AbstractString}} = nothing)
    io = IOBuffer()
    md = rep.metadata
    println(io, "# RCA report")
    println(io)
    println(io, "- lines: **", md["n_lines"], "**")
    println(io, "- clusters: **", md["n_clusters"], "**")
    println(io, "- anomaly cutoff (percentile ",
            round(md["top_percentile"]; digits = 3), "): **",
            round(md["threshold"]; digits = 4), "**")
    println(io, "- anomalous lines: **", md["n_anomalies"], "**")
    println(io, "- episode knobs: min_sup=", md["min_sup"],
            ", max_gap=", md["max_gap"],
            ", max_time_duration=", md["max_time_duration"])
    println(io, "- detector used: ", md["with_detector"])
    println(io)
    println(io, "## Top ", min(Int(topk), length(rep.ranked)),
            " root-cause episodes")
    println(io)
    println(io, "| # | pattern (cluster-id seq) | support | density | score |")
    println(io, "|---|---|---:|---:|---:|")
    for (i, r) in enumerate(rep.ranked)
        i > topk && break
        println(io, "| ", i, " | `", r.pattern, "` | ", r.support,
                " | ", round(r.density; digits = 4),
                " | ", round(r.score; digits = 2), " |")
    end
    if lines !== nothing && !isempty(rep.ranked)
        println(io)
        println(io, "## Representative lines (first occurrence of each of top ",
                min(Int(topk), length(rep.ranked)), ")")
        for (i, r) in enumerate(rep.ranked)
            i > topk && break
            occ = rep.episodes[r.pattern]
            isempty(occ) && continue
            first_positions = first(occ)
            println(io)
            println(io, "### #", i, " pattern `", r.pattern, "`")
            for pos in first_positions
                1 <= pos <= length(lines) || continue
                println(io, "- L", pos, ": `", lines[pos], "`")
            end
        end
    end
    return String(take!(io))
end

end # module RCA
