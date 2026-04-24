"""
    Featurise

Raw-line → numeric-feature pipeline. Turns a vector of log lines into
a `Vocabulary` + a feature matrix the Lux model factories accept
directly.

- [`Vocabulary`] — token ↔ id bijection with a reserved `<UNK>` at
  id `1`.
- [`build_vocab`] — scan a corpus and emit a `Vocabulary` with
  optional `min_count` / `max_vocab` cutoffs; masks typed slots
  first by default (so `10.0.0.1` and `10.0.0.2` share one
  `<IP>` token, not two).
- [`tokenise`] / [`tokenise_ids`] — one line → tokens / ids.
- [`bow`] — full corpus → `(vocab_size, n_lines)` `Float32`
  term-count or log-frequency matrix (default log: the thesis's
  `normalize_log` signal).
- [`sequence_matrix`] — full corpus → `(seqlen, n_lines)` `Int`
  id matrix with left-padding / truncation, suitable as
  `seq_lstm` input.

Design note: every featurisation step is deterministic, pure, and
records enough state in the returned `Vocabulary` that a later
`classify` pass on new lines produces the same column ordering.
That's what lets `Persistence.save_deep_kate(…; vocab = vocab)` +
`classify --model path.jld2 --data newlog.log` work.
"""
module Featurise

using ..Masking: mask_lines

export Vocabulary, build_vocab, tokenise, tokenise_ids,
       bow, sequence_matrix,
       build_ngram_profile, NGramProfile

"""
    Vocabulary(tokens, index)

Token ↔ 1-based integer id. `tokens[1]` is always `"<UNK>"` — any
out-of-vocabulary token maps to id `1`.
"""
struct Vocabulary
    tokens::Vector{String}
    index::Dict{String, Int}
    mask::Bool            # whether the vocab was built against
                          # typed-slot-masked tokens (if so, all
                          # downstream lines go through the same
                          # Masking.mask_lines step before tokenising)
end

Base.length(v::Vocabulary) = length(v.tokens)
Base.show(io::IO, v::Vocabulary) = print(io,
    "Vocabulary(", length(v), " tokens",
    v.mask ? "; masked" : "", ")")

"""
    tokenise(line::AbstractString) -> Vector{SubString}

Whitespace tokeniser — the thesis keeps separators as their own
tokens downstream via the regex-cascade parser, but for BoW we
split on `\\s+` and drop the separators.
"""
tokenise(line::AbstractString) = split(line, r"\s+"; keepempty = false)

# ---------------------------------------------------------------------------
# Char n-gram similarity infrastructure (inference-time OOV routing)
# ---------------------------------------------------------------------------

"""
    NGramProfile

Per-token char n-gram fingerprint built once from a
[`Vocabulary`] and reused across every call to `tokenise_ids` /
`bow` with `oov_policy ∈ (:nearest, :distribute)`. `grams[id]` is
the set of n-grams for `vocab.tokens[id]` (`id ≥ 2` — the UNK
slot is skipped).
"""
struct NGramProfile
    grams::Vector{Set{String}}
    n::Int
end

function _char_ngrams(token::AbstractString, n::Int)
    # Bracketed with `<` / `>` so `<ab` and `ab>` are prefix / suffix
    # n-grams distinct from any internal `ab` that may land in the middle
    # of a longer token.
    padded = "<" * token * ">"
    out = Set{String}()
    len = length(padded)
    if len < n
        push!(out, padded)
        return out
    end
    chars = collect(padded)
    @inbounds for i in 1:(len - n + 1)
        push!(out, String(chars[i:i + n - 1]))
    end
    return out
end

"""
    build_ngram_profile(vocab; n = 3) -> NGramProfile

Precompute a char-`n`-gram profile for every token in `vocab`
(skipping `<UNK>` at id 1). Tokens are bracketed with `<` / `>`
sentinels so prefix and suffix n-grams don't collide with inner
ones. Empty or too-short tokens produce an empty n-gram set.
"""
function build_ngram_profile(vocab::Vocabulary; n::Integer = 3)
    nn = Int(n)
    nn >= 1 || throw(ArgumentError("n must be ≥ 1"))
    grams = Vector{Set{String}}(undef, length(vocab))
    @inbounds for i in eachindex(vocab.tokens)
        grams[i] = i == 1 ? Set{String}() : _char_ngrams(vocab.tokens[i], nn)
    end
    return NGramProfile(grams, nn)
end

@inline function _jaccard(a::Set{String}, b::Set{String})
    (isempty(a) && isempty(b)) && return 0.0
    inter = length(intersect(a, b))
    union_ = length(a) + length(b) - inter
    return union_ == 0 ? 0.0 : Float64(inter) / Float64(union_)
end

"""
    _nearest_token(oov, vocab, profile) -> (id::Int, sim::Float64)

Best matching vocab token (by char-`n`-gram Jaccard) for `oov`.
Ties broken by shorter vocab-token length, then by lower id. The
UNK slot (id 1) is never considered. Returns `(1, 0.0)` if the
vocab is empty or every similarity is zero.
"""
function _nearest_token(oov::AbstractString, vocab::Vocabulary,
                        profile::NGramProfile)
    q = _char_ngrams(String(oov), profile.n)
    best_id = 1
    best_sim = 0.0
    best_len = typemax(Int)
    @inbounds for id in 2:length(vocab)
        sim = _jaccard(q, profile.grams[id])
        improved = sim > best_sim ||
                   (sim == best_sim && length(vocab.tokens[id]) < best_len)
        improved || continue
        best_id = id
        best_sim = sim
        best_len = length(vocab.tokens[id])
    end
    return best_id, best_sim
end

"""
    _topk_token_sims(oov, vocab, profile, k) -> Vector{Pair{Int, Float64}}

Top-`k` `(id, similarity)` pairs by char-n-gram Jaccard, sorted
descending. Used by `bow`'s `:distribute` path.
"""
function _topk_token_sims(oov::AbstractString, vocab::Vocabulary,
                          profile::NGramProfile, k::Integer)
    q = _char_ngrams(String(oov), profile.n)
    sims = Tuple{Int, Float64}[]
    @inbounds for id in 2:length(vocab)
        sim = _jaccard(q, profile.grams[id])
        sim > 0 && push!(sims, (id, sim))
    end
    isempty(sims) && return Pair{Int, Float64}[]
    sort!(sims; by = t -> t[2], rev = true)
    kk = min(Int(k), length(sims))
    return [Pair{Int, Float64}(id, sim) for (id, sim) in sims[1:kk]]
end

"""
    tokenise_ids(line, vocab;
                 oov_policy = :unk,
                 profile = nothing,
                 oov_min_sim = 0.3,
                 oov_top_k = 3) -> Vector{Int}

Tokenise `line` and map each token to its vocab id. When a token
isn't in `vocab.index` the `oov_policy` decides where it goes:

- `:unk` (default) — route to id `1`, the reserved `<UNK>` slot.
  Matches the pre-existing behaviour.
- `:nearest` — compute char-3-gram Jaccard similarity between the
  OOV token and every vocab token using the supplied or
  lazily-built `profile`; if the best similarity is ≥ `oov_min_sim`
  (default 0.3) route to that token, else fall through to `:unk`.
  This is the inference-time fix for corpora whose new
  identifiers look lexically like known ones
  (`"prod-gpu-042"` ↔ `"staging-gpu-017"`).

For the soft `:distribute` variant (fractional routing across
top-k similar vocab tokens) use [`bow`] with `oov_policy =
:distribute` — this integer-only entry point can't represent
fractional contributions.

`profile` is a [`NGramProfile`] built via
[`build_ngram_profile`]; pass it explicitly when calling
`tokenise_ids` in a loop so the char-3-gram index is computed
once per vocab, not once per line. When omitted, a profile is
constructed on demand (slow for repeated calls).

If `vocab.mask == true`, the line is run through
`Masking.mask_line` first so typed slots share ids with training.
"""
function tokenise_ids(line::AbstractString, vocab::Vocabulary;
                      oov_policy::Symbol = :unk,
                      profile::Union{Nothing, NGramProfile} = nothing,
                      oov_min_sim::Real = 0.3,
                      oov_top_k::Integer = 3)
    L = vocab.mask ? first(mask_lines([line])) : line
    ids = Int[]
    if oov_policy === :unk
        for t in tokenise(L)
            push!(ids, get(vocab.index, String(t), 1))
        end
        return ids
    elseif oov_policy === :nearest
        p = profile === nothing ? build_ngram_profile(vocab) : profile
        for t in tokenise(L)
            tok = String(t)
            id = get(vocab.index, tok, 0)
            if id > 0
                push!(ids, id)
            else
                best_id, best_sim = _nearest_token(tok, vocab, p)
                push!(ids, best_sim >= oov_min_sim ? best_id : 1)
            end
        end
        return ids
    elseif oov_policy === :distribute
        throw(ArgumentError(":distribute is only supported by `bow`; " *
                            "call `bow(lines, vocab; oov_policy = :distribute)` " *
                            "to get fractional counts."))
    else
        throw(ArgumentError("unknown oov_policy `$oov_policy`; " *
                            "use :unk, :nearest, or :distribute"))
    end
end

# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------

"""
    build_vocab(lines; mask = true, min_count = 1,
                max_vocab = typemax(Int), unk_token = "<UNK>")

Scan `lines` and return a [`Vocabulary`]. Tokens are counted
case-sensitively; the `min_count` and `max_vocab` knobs drop
low-frequency noise and cap the final matrix width.

When `mask = true` (default) the corpus is first passed through
`Masking.mask_lines`, so variable values (`<IP>`, `<INT>`, …) share
a single vocabulary slot per type. Disable with `mask = false` to
tokenise raw lines verbatim — slower to plateau but occasionally
useful when the typed slots are themselves the signal.
"""
function build_vocab(lines::AbstractVector{<:AbstractString};
                     mask::Bool = true,
                     min_count::Integer = 1,
                     max_vocab::Integer = typemax(Int),
                     unk_token::AbstractString = "<UNK>")
    min_count >= 1 || throw(ArgumentError("min_count must be ≥ 1"))
    max_vocab >= 1 || throw(ArgumentError("max_vocab must be ≥ 1"))
    prepared = mask ? mask_lines(lines) : collect(String, lines)
    counts = Dict{String, Int}()
    for l in prepared
        for t in tokenise(l)
            counts[String(t)] = get(counts, String(t), 0) + 1
        end
    end
    # Sort by frequency descending, break ties lexicographically
    # so the vocab is deterministic across runs.
    pairs = [(k, v) for (k, v) in counts if v >= min_count]
    sort!(pairs; by = p -> (-p[2], p[1]))
    # UNK at id 1; then at most `max_vocab - 1` real tokens.
    tokens = String[String(unk_token)]
    keep = min(length(pairs), max_vocab - 1)
    for i in 1:keep
        push!(tokens, pairs[i][1])
    end
    index = Dict{String, Int}(t => i for (i, t) in enumerate(tokens))
    return Vocabulary(tokens, index, mask)
end

# ---------------------------------------------------------------------------
# BoW matrix
# ---------------------------------------------------------------------------

"""
    bow(lines, vocab; normalise = :log, eltype = Float32)
        -> Matrix{eltype}

Bag-of-words feature matrix. Output shape is
`(length(vocab), length(lines))` so the `(features, samples)`
convention used across the Lux factories applies unchanged.

`normalise`:

- `:count` — raw term-frequency counts.
- `:log`   — `log(1 + count) / log(vocab_size) · log(1 + count)`,
  the thesis's `normalize_log` (`src/KATE.jl`). Default.
- `:l1`    — divide each column by its own sum (turns counts into
  a probability vector per line).
- `:binary`— `x[i, j] = 1` iff token `i` appeared in line `j`.

The `:log` form is what KATE / DeepKATE were designed for; use
`:binary` when feeding VQ-VAE or SimCSE and `:l1` for a
plain multinomial-style AE baseline.
"""
function bow(lines::AbstractVector{<:AbstractString},
             vocab::Vocabulary;
             normalise::Symbol = :log,
             oov_policy::Symbol = :unk,
             oov_min_sim::Real = 0.3,
             oov_top_k::Integer = 3,
             eltype::Type{T} = Float32) where {T}
    V = length(vocab)
    N = length(lines)
    X = zeros(T, V, N)
    profile = oov_policy === :unk ? nothing : build_ngram_profile(vocab)
    min_sim = Float64(oov_min_sim)
    for (j, l) in enumerate(lines)
        L = vocab.mask ? first(mask_lines([String(l)])) : l
        for t in tokenise(L)
            tok = String(t)
            id = get(vocab.index, tok, 0)
            if id > 0
                @inbounds X[id, j] += one(T)
            elseif oov_policy === :unk
                @inbounds X[1, j] += one(T)
            elseif oov_policy === :nearest
                best_id, best_sim = _nearest_token(tok, vocab, profile)
                target = best_sim >= min_sim ? best_id : 1
                @inbounds X[target, j] += one(T)
            elseif oov_policy === :distribute
                sims = _topk_token_sims(tok, vocab, profile, oov_top_k)
                if isempty(sims) || first(sims).second < min_sim
                    @inbounds X[1, j] += one(T)
                else
                    total = sum(p -> p.second, sims)
                    @inbounds for (id_, sim) in sims
                        X[id_, j] += T(sim / total)
                    end
                end
            else
                throw(ArgumentError("unknown oov_policy `$oov_policy`; " *
                                    "use :unk, :nearest, or :distribute"))
            end
        end
    end
    return _normalise!(X, normalise)
end

function _normalise!(X::AbstractMatrix{T}, kind::Symbol) where {T}
    if kind === :count
        return X
    elseif kind === :log
        # Matches the thesis's normalize_log exactly but over the
        # feature axis rather than a whole-document term-count dict.
        V = size(X, 1)
        V <= 1 && return X
        logV = log(T(V))
        @inbounds for j in axes(X, 2), i in axes(X, 1)
            c = X[i, j]
            if c > 0
                l = log(T(1) + c)
                X[i, j] = l / logV * l
            end
        end
        return X
    elseif kind === :l1
        @inbounds for j in axes(X, 2)
            s = sum(@view X[:, j])
            s > 0 && (@view(X[:, j]) ./= s)
        end
        return X
    elseif kind === :binary
        @inbounds for i in eachindex(X)
            X[i] = X[i] > 0 ? one(T) : zero(T)
        end
        return X
    else
        throw(ArgumentError("unknown normalise=$(kind); " *
                            "use :count, :log, :l1, or :binary"))
    end
end

# ---------------------------------------------------------------------------
# Sequence matrix (SeqLSTM input)
# ---------------------------------------------------------------------------

"""
    sequence_matrix(lines, vocab; seqlen = 16, pad_id = 1) -> Matrix{Int}

Map each line to a length-`seqlen` Int id sequence and stack them
column-wise into a `(seqlen, length(lines))` matrix. Lines longer
than `seqlen` are **truncated on the right**; shorter lines are
**left-padded with `pad_id`** (which defaults to the `<UNK>` id).
Suitable as input to `SeqLSTM.seq_lstm(vocab_size; …)`.
"""
function sequence_matrix(lines::AbstractVector{<:AbstractString},
                         vocab::Vocabulary;
                         seqlen::Integer = 16,
                         pad_id::Integer = 1,
                         oov_policy::Symbol = :unk,
                         oov_min_sim::Real = 0.3)
    seqlen >= 1 || throw(ArgumentError("seqlen must be ≥ 1"))
    oov_policy in (:unk, :nearest) ||
        throw(ArgumentError("sequence_matrix supports oov_policy in " *
                            "(:unk, :nearest); :distribute is BoW-only"))
    X = fill(Int(pad_id), Int(seqlen), length(lines))
    profile = oov_policy === :unk ? nothing : build_ngram_profile(vocab)
    for (j, l) in enumerate(lines)
        ids = tokenise_ids(l, vocab;
                           oov_policy = oov_policy,
                           profile = profile,
                           oov_min_sim = oov_min_sim)
        k = min(length(ids), seqlen)
        @inbounds for i in 1:k
            # Right-align so left-padding survives the model's
            # temporal order (the last position holds the most
            # recent token — canonical for next-token prediction).
            X[seqlen - k + i, j] = ids[i]
        end
    end
    return X
end

end # module Featurise
