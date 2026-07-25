"""
    PostProc.Merge

Cluster-template merge — plan 001 Stage E′ post-processing step 4.

Drain (and any tree-routed parser) over-splits when a long-tail
token routes a line to a sibling leaf. Once parsing is done, two
clusters whose *canonical templates* are within a small token-edit
distance — and (optionally) whose embedding centroids are close —
should be merged.

This module supplies the deterministic half: token-level edit
distance + a guarded merge that respects `max_token_distance`
and `max_wildcard_diff`. The cosine-on-embedding check is left to
the caller because the embedder choice is application-specific —
just pass `centroids` if you have them, otherwise the merge
proceeds purely from the templates.
"""
module Merge

using ..Canonical: canonicalise

export token_edit_distance, merge_clusters, merge_pairs

"""
    token_edit_distance(a, b) -> Int

Levenshtein edit distance over whitespace-tokenised strings. Used by
[`merge_pairs`] to find candidate cluster pairs. Symmetric, returns
`0` iff the two templates tokenise identically.
"""
function token_edit_distance(a::AbstractString, b::AbstractString)
    ta = collect(split(a))
    tb = collect(split(b))
    m, n = length(ta), length(tb)
    m == 0 && return n
    n == 0 && return m
    prev = collect(0:n)
    curr = zeros(Int, n + 1)
    @inbounds for i in 1:m
        curr[1] = i
        for j in 1:n
            cost = ta[i] == tb[j] ? 0 : 1
            curr[j + 1] = min(prev[j + 1] + 1,    # delete
                              curr[j] + 1,        # insert
                              prev[j] + cost)     # substitute
        end
        prev, curr = curr, prev
    end
    return prev[n + 1]
end

"""
    merge_pairs(templates; max_token_distance = 1, max_wildcard_diff = 1)
        -> Vector{Tuple{Int, Int}}

Identify pairs `(i, j)` with `i < j` whose canonical templates
differ by ≤ `max_token_distance` tokens and whose wildcard counts
differ by ≤ `max_wildcard_diff`. Both guards keep the merge
conservative — small string drift only.
"""
function merge_pairs(templates::AbstractVector{<:AbstractString};
                     max_token_distance::Integer = 1,
                     max_wildcard_diff::Integer = 1,
                     wildcard::AbstractString = "<*>")
    n = length(templates)
    canon = String[canonicalise(t; wildcard = wildcard) for t in templates]
    pairs = Tuple{Int, Int}[]
    @inbounds for i in 1:n - 1, j in i + 1:n
        ed = token_edit_distance(canon[i], canon[j])
        ed > max_token_distance && continue
        wi = count(==(wildcard), split(canon[i]))
        wj = count(==(wildcard), split(canon[j]))
        abs(wi - wj) > max_wildcard_diff && continue
        push!(pairs, (i, j))
    end
    return pairs
end

"""
    merge_clusters(assignments::AbstractVector{<:Integer},
                   templates::AbstractVector{<:AbstractString};
                   max_token_distance = 1, max_wildcard_diff = 1)
        -> (new_assignments::Vector{Int},
            new_templates::Vector{String})

Apply [`merge_pairs`] and remap `assignments` so lines that landed
in pair-merged clusters share the same id. The surviving template
of each merged group is the *lexicographically smallest* canonical
form (deterministic; not optimised for "best" template — that's
dataset-dependent).

The result is suitable as a drop-in replacement for the parser's
own assignments / templates ahead of an evaluation pass.
"""
function merge_clusters(assignments::AbstractVector{<:Integer},
                        templates::AbstractVector{<:AbstractString};
                        max_token_distance::Integer = 1,
                        max_wildcard_diff::Integer = 1,
                        wildcard::AbstractString = "<*>")
    length(assignments) == length(templates) ||
        throw(DimensionMismatch("assignments and templates must agree in length"))
    # Work over the *unique* templates so the O(n²) pair search
    # doesn't see duplicates.
    uniq_templates = String[]
    canonical_of = Dict{Int, String}()
    template_id = Dict{String, Int}()
    line_template_id = Vector{Int}(undef, length(templates))
    @inbounds for (i, t) in enumerate(templates)
        c = canonicalise(t; wildcard = wildcard)
        if !haskey(template_id, c)
            push!(uniq_templates, c)
            template_id[c] = length(uniq_templates)
        end
        line_template_id[i] = template_id[c]
    end
    pairs = merge_pairs(uniq_templates;
                        max_token_distance = max_token_distance,
                        max_wildcard_diff = max_wildcard_diff,
                        wildcard = wildcard)
    # Union-find over template-ids.
    parent = collect(1:length(uniq_templates))
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb && return
        # Pin the canonical/lower-id root for determinism.
        if ra < rb
            parent[rb] = ra
        else
            parent[ra] = rb
        end
    end
    for (i, j) in pairs
        union!(i, j)
    end
    # Choose surviving template per group: lex-smallest canonical.
    rep_template = Dict{Int, String}()
    @inbounds for tid in 1:length(uniq_templates)
        root = find(tid)
        cand = uniq_templates[tid]
        if !haskey(rep_template, root) || cand < rep_template[root]
            rep_template[root] = cand
        end
    end
    # Re-key assignments by the surviving template's index.
    surviving_id = Dict{String, Int}()
    new_templates = String[]
    @inbounds for tid in 1:length(uniq_templates)
        root = find(tid)
        tpl = rep_template[root]
        if !haskey(surviving_id, tpl)
            push!(new_templates, tpl)
            surviving_id[tpl] = length(new_templates)
        end
    end
    new_assignments = Vector{Int}(undef, length(assignments))
    @inbounds for i in eachindex(assignments)
        tpl = rep_template[find(line_template_id[i])]
        new_assignments[i] = surviving_id[tpl]
    end
    return new_assignments, new_templates
end

end # module Merge
