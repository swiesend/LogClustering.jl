"""
    Eval.Metrics

Pure-Julia parsing and clustering metrics for plan 001 Stage F.

**Parsing metrics** (Zhu et al. ISSRE 2023, LogHub-2.0):
- [`parsing_accuracy`] — fraction of log lines whose *inferred*
  template string equals the *ground-truth* template string.
- [`group_accuracy`] — fraction of log lines whose inferred group
  equals its ground-truth group.
- [`template_group_f1`] — template-level precision / recall / F1; the
  "FTA" of LogHub-2.0.
- [`grouping_f1`] — pair-based grouping precision / recall / F1; the
  "FGA" of LogHub-2.0.

**Clustering metrics** (standard):
- [`normalised_mutual_information`] ([`nmi`])
- [`adjusted_rand_index`] ([`ari`])
- [`purity`]
- [`v_measure`]

All of these work on integer label vectors of equal length; conversion
from string templates to integer ids is handled by [`to_labels`].
"""
module Metrics

using Statistics

export to_labels,
       parsing_accuracy, group_accuracy,
       template_group_f1, grouping_f1,
       normalised_mutual_information, nmi,
       adjusted_rand_index, ari,
       purity, v_measure

# ---------------------------------------------------------------------------
# String-template → integer label
# ---------------------------------------------------------------------------

"""
    to_labels(strings) -> Vector{Int}

Map a vector of string templates (or event ids) to a `Vector{Int}`
whose entries index into `unique(strings)`, stable in first-seen order.
Two equal strings map to the same integer.
"""
function to_labels(v::AbstractVector{<:AbstractString})
    d = Dict{String, Int}()
    out = Vector{Int}(undef, length(v))
    @inbounds for (i, s) in enumerate(v)
        s = String(s)
        out[i] = get!(d, s) do
            length(d) + 1
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Parsing metrics (LogHub-2.0)
# ---------------------------------------------------------------------------

"""
    parsing_accuracy(pred_templates, gold_templates) -> Float64

Fraction of lines where the inferred template *string* equals the
ground-truth template *string*.
"""
function parsing_accuracy(pred::AbstractVector{<:AbstractString},
                          gold::AbstractVector{<:AbstractString})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    isempty(pred) && return 0.0
    ok = 0
    @inbounds for i in eachindex(pred)
        pred[i] == gold[i] && (ok += 1)
    end
    return ok / length(pred)
end

"""
    group_accuracy(pred_labels, gold_labels) -> Float64

Fraction of lines `i` where the multiset `{j : pred[j] == pred[i]}`
equals `{j : gold[j] == gold[i]}`. Equivalent to LogHub's "GA".
"""
function group_accuracy(pred::AbstractVector{<:Integer},
                        gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    isempty(pred) && return 0.0
    pred_groups = _group_indices(pred)
    gold_groups = _group_indices(gold)
    ok = 0
    @inbounds for i in eachindex(pred)
        pred_groups[pred[i]] == gold_groups[gold[i]] && (ok += 1)
    end
    return ok / length(pred)
end

group_accuracy(pred::AbstractVector{<:AbstractString},
               gold::AbstractVector{<:AbstractString}) =
    group_accuracy(to_labels(pred), to_labels(gold))

"""
    grouping_f1(pred_labels, gold_labels) -> (precision, recall, f1)

Pair-based grouping F1 (LogHub "FGA"). Counts pairs `(i, j)` with
`i < j`; TP = same cluster in both, FP = same in pred but not gold,
FN = same in gold but not pred.
"""
function grouping_f1(pred::AbstractVector{<:Integer},
                     gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    tp = 0
    fp = 0
    fn = 0
    n = length(pred)
    @inbounds for i in 1:n-1, j in i+1:n
        same_pred = pred[i] == pred[j]
        same_gold = gold[i] == gold[j]
        if same_pred && same_gold
            tp += 1
        elseif same_pred && !same_gold
            fp += 1
        elseif !same_pred && same_gold
            fn += 1
        end
    end
    prec = tp + fp == 0 ? 0.0 : tp / (tp + fp)
    rec  = tp + fn == 0 ? 0.0 : tp / (tp + fn)
    f1   = prec + rec == 0 ? 0.0 : 2prec * rec / (prec + rec)
    return prec, rec, f1
end

grouping_f1(pred::AbstractVector{<:AbstractString},
            gold::AbstractVector{<:AbstractString}) =
    grouping_f1(to_labels(pred), to_labels(gold))

"""
    template_group_f1(pred_templates, gold_templates) -> (precision, recall, f1)

Template-level F1 (LogHub "FTA"). A template is "correct" if the set
of lines it covers exactly matches a ground-truth template's line set.
- Precision = correct / |predicted templates|
- Recall    = correct / |ground-truth templates|
"""
function template_group_f1(pred::AbstractVector{<:Integer},
                           gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    pred_sets = _label_sets(pred)
    gold_sets = _label_sets(gold)
    gold_set_of_sets = Set(values(gold_sets))
    correct = 0
    for s in values(pred_sets)
        s in gold_set_of_sets && (correct += 1)
    end
    prec = isempty(pred_sets) ? 0.0 : correct / length(pred_sets)
    rec  = isempty(gold_sets) ? 0.0 : correct / length(gold_sets)
    f1   = prec + rec == 0 ? 0.0 : 2prec * rec / (prec + rec)
    return prec, rec, f1
end

template_group_f1(pred::AbstractVector{<:AbstractString},
                  gold::AbstractVector{<:AbstractString}) =
    template_group_f1(to_labels(pred), to_labels(gold))

# ---------------------------------------------------------------------------
# Clustering metrics (pure Julia)
# ---------------------------------------------------------------------------

"""
    purity(pred, gold) -> Float64

`1/N · Σ max_k |cluster_i ∩ class_k|`. Not symmetric — `pred` is the
clustering to evaluate, `gold` is the reference labelling.
"""
function purity(pred::AbstractVector{<:Integer}, gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    N = length(pred)
    N == 0 && return 0.0
    cm = _contingency(pred, gold)
    total = 0
    for row in eachrow(cm)
        total += maximum(row)
    end
    return total / N
end

purity(pred::AbstractVector{<:AbstractString}, gold::AbstractVector{<:AbstractString}) =
    purity(to_labels(pred), to_labels(gold))

"""
    normalised_mutual_information(pred, gold) -> Float64

Symmetric NMI: `MI(pred, gold) / √(H(pred) · H(gold))`. Returns 0 when
either side is constant (entropy 0).
"""
function normalised_mutual_information(pred::AbstractVector{<:Integer},
                                       gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    N = length(pred)
    N == 0 && return 0.0
    cm = _contingency(pred, gold)
    row_sums = sum(cm; dims = 2)
    col_sums = sum(cm; dims = 1)
    H_pred = _entropy(vec(row_sums), N)
    H_gold = _entropy(vec(col_sums), N)
    (H_pred == 0 || H_gold == 0) && return 0.0
    mi = 0.0
    @inbounds for i in axes(cm, 1), j in axes(cm, 2)
        nij = cm[i, j]
        nij == 0 && continue
        pij = nij / N
        mi += pij * log(nij * N / (row_sums[i, 1] * col_sums[1, j]))
    end
    return mi / sqrt(H_pred * H_gold)
end

const nmi = normalised_mutual_information

nmi(pred::AbstractVector{<:AbstractString}, gold::AbstractVector{<:AbstractString}) =
    nmi(to_labels(pred), to_labels(gold))

"""
    v_measure(pred, gold; β = 1.0) -> (homogeneity, completeness, V)

Rosenberg & Hirschberg 2007. `β = 1` weights homogeneity and
completeness equally.
"""
function v_measure(pred::AbstractVector{<:Integer}, gold::AbstractVector{<:Integer};
                   β::Real = 1.0)
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    N = length(pred)
    N == 0 && return 0.0, 0.0, 0.0
    cm = _contingency(pred, gold)
    row_sums = vec(sum(cm; dims = 2))
    col_sums = vec(sum(cm; dims = 1))
    H_pred = _entropy(row_sums, N)
    H_gold = _entropy(col_sums, N)
    H_pred_given_gold = _conditional_entropy(cm, col_sums, N)
    H_gold_given_pred = _conditional_entropy(permutedims(cm), row_sums, N)
    homogeneity   = H_gold == 0 ? 1.0 : 1 - H_gold_given_pred / H_gold
    completeness  = H_pred == 0 ? 1.0 : 1 - H_pred_given_gold / H_pred
    denom = β * homogeneity + completeness
    V = denom == 0 ? 0.0 : (1 + β) * homogeneity * completeness / denom
    return homogeneity, completeness, V
end

v_measure(pred::AbstractVector{<:AbstractString}, gold::AbstractVector{<:AbstractString};
          kwargs...) = v_measure(to_labels(pred), to_labels(gold); kwargs...)

"""
    adjusted_rand_index(pred, gold) -> Float64

Hubert & Arabie 1985 ARI. Returns `(RI − E[RI]) / (max RI − E[RI])`.
"""
function adjusted_rand_index(pred::AbstractVector{<:Integer},
                             gold::AbstractVector{<:Integer})
    length(pred) == length(gold) || throw(DimensionMismatch("pred vs gold length"))
    N = length(pred)
    N < 2 && return 0.0
    cm = _contingency(pred, gold)
    row_sums = vec(sum(cm; dims = 2))
    col_sums = vec(sum(cm; dims = 1))
    c2(n) = n * (n - 1) ÷ 2
    index = sum(c2(cm[i, j]) for i in axes(cm, 1), j in axes(cm, 2))
    a = sum(c2.(row_sums))
    b = sum(c2.(col_sums))
    total = c2(N)
    expected = total == 0 ? 0.0 : a * b / total
    max_index = (a + b) / 2
    denom = max_index - expected
    return denom == 0 ? 0.0 : (index - expected) / denom
end

const ari = adjusted_rand_index

ari(pred::AbstractVector{<:AbstractString}, gold::AbstractVector{<:AbstractString}) =
    ari(to_labels(pred), to_labels(gold))

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

function _contingency(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    ka = maximum(a)
    kb = maximum(b)
    cm = zeros(Int, ka, kb)
    @inbounds for i in eachindex(a)
        cm[a[i], b[i]] += 1
    end
    return cm
end

"Return a `Dict{label, Set{index}}` for every label in `v`."
function _label_sets(v::AbstractVector{<:Integer})
    out = Dict{Int, Set{Int}}()
    @inbounds for (i, l) in enumerate(v)
        push!(get!(out, l, Set{Int}()), i)
    end
    return out
end

"For each line, the *set of indices* that share its label — O(N)."
function _group_indices(v::AbstractVector{<:Integer})
    sets = _label_sets(v)
    return sets
end

function _entropy(counts::AbstractVector{<:Integer}, N::Integer)
    h = 0.0
    @inbounds for c in counts
        c == 0 && continue
        p = c / N
        h -= p * log(p)
    end
    return h
end

function _conditional_entropy(cm::AbstractMatrix{<:Integer},
                              col_sums::AbstractVector{<:Integer}, N::Integer)
    # H(rows | cols) = Σ_j p(col_j) · H(rows | col_j)
    h = 0.0
    @inbounds for j in axes(cm, 2)
        col = col_sums[j]
        col == 0 && continue
        pj = col / N
        hj = 0.0
        for i in axes(cm, 1)
            nij = cm[i, j]
            nij == 0 && continue
            p = nij / col
            hj -= p * log(p)
        end
        h += pj * hj
    end
    return h
end

end # module Metrics
