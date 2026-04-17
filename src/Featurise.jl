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
       bow, sequence_matrix

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

"""
    tokenise_ids(line, vocab) -> Vector{Int}

Tokenise `line` and map each token to its vocab id, substituting
`1` (the `<UNK>` slot) for anything not in `vocab.index`. If
`vocab.mask == true`, runs through `Masking.mask_line` first so a
new line's typed slots share the same token ids as the training
corpus's did.
"""
function tokenise_ids(line::AbstractString, vocab::Vocabulary)
    L = vocab.mask ? first(mask_lines([line])) : line
    ids = Int[]
    for t in tokenise(L)
        push!(ids, get(vocab.index, String(t), 1))
    end
    return ids
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
             eltype::Type{T} = Float32) where {T}
    V = length(vocab)
    N = length(lines)
    X = zeros(T, V, N)
    for (j, l) in enumerate(lines)
        ids = tokenise_ids(l, vocab)
        @inbounds for id in ids
            X[id, j] += one(T)
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
                         pad_id::Integer = 1)
    seqlen >= 1 || throw(ArgumentError("seqlen must be ≥ 1"))
    X = fill(Int(pad_id), Int(seqlen), length(lines))
    for (j, l) in enumerate(lines)
        ids = tokenise_ids(l, vocab)
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
