module KATE

using Random: AbstractRNG
using LuxCore: LuxCore, AbstractLuxLayer
using WeightInitializers: glorot_uniform, zeros32
using ChainRulesCore: @ignore_derivatives

export KCompetetive, count_words, normalize_log, transform_text_to_input, get_similar_words

"""
    KCompetetive(in, out, σ=tanh, α=6.26; k=out, init_weight=glorot_uniform, init_bias=zeros32)

K-competitive autoencoder layer (Chen & Zaki, KDD 2017).

Computes the pre-activation `h = W*x + b` (size `out`), applies
k-winner-take-all competition separately on the positive and negative
halves of `h` (keeping `k/2` winners each), redistributes the "energy"
of the losers to the winners (scaled by `α`), zeroes the losers, and
finally applies `σ`.

`k` is the number of competitive winners (not the output dimension).
The DeepKATE topology (thesis §3.2.3) uses `k < out` so most neurons
become losers; passing only positional arguments preserves the
original-KATE default `k = out`. `k` must be `≤ out`. Odd `k` is
allowed — `⌊k/2⌋` winners are kept on the positive and negative side
each, so one slot is unused when `k` is odd.
"""
struct KCompetetive{F, IW, IB} <: AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    k::Int
    activation::F
    alpha::Float32
    init_weight::IW
    init_bias::IB
end

function KCompetetive(in::Integer, out::Integer, σ = tanh, α::Real = 6.26f0;
                      k::Integer = out,
                      init_weight = glorot_uniform, init_bias = zeros32)
    k > in && throw(ArgumentError("k ($k) must not exceed in ($in)"))
    k > out && throw(ArgumentError("k ($k) must not exceed out ($out)"))
    k < 1 && throw(ArgumentError("k ($k) must be ≥ 1"))
    return KCompetetive(Int(in), Int(out), Int(k), σ, Float32(α), init_weight, init_bias)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::KCompetetive)
    return (weight = l.init_weight(rng, l.out_dims, l.in_dims),
            bias   = l.init_bias(rng, l.out_dims))
end

LuxCore.initialstates(::AbstractRNG, ::KCompetetive) = NamedTuple()
LuxCore.parameterlength(l::KCompetetive) = l.out_dims * l.in_dims + l.out_dims
LuxCore.statelength(::KCompetetive) = 0

function Base.show(io::IO, l::KCompetetive)
    print(io, "KCompetetive(", l.in_dims, " => ", l.out_dims)
    l.activation === identity || print(io, ", ", l.activation)
    print(io, "; k=", l.k, ", α=", l.alpha, ")")
end

"""
    _kcomp_indices(h, k)

Non-differentiable helper: partition hidden pre-activations `h` into
positive/negative winners and losers. Returns index vectors
`(pos_winners, pos_losers, neg_winners, neg_losers)`.
"""
function _kcomp_indices(h::AbstractVector, k::Integer)
    half = k ÷ 2
    pos = Int[]
    neg = Int[]
    @inbounds for i in eachindex(h)
        h[i] >= 0 ? push!(pos, i) : push!(neg, i)
    end
    # ascending by value: smallest positives are losers
    sort!(pos; by = i -> h[i])
    # descending by value: values closest to 0 (least negative) are losers
    sort!(neg; by = i -> h[i], rev = true)

    P, N = length(pos), length(neg)
    p_losers = P > half ? pos[1:P-half]      : Int[]
    p_winners = P > half ? pos[P-half+1:end] : pos
    n_losers = N > half ? neg[1:N-half]      : Int[]
    n_winners = N > half ? neg[N-half+1:end] : neg
    return p_winners, p_losers, n_winners, n_losers
end

function _kcomp_mask(len::Int, idxs::Vector{Int}, ::Type{T}) where {T}
    m = zeros(T, len)
    @inbounds for i in idxs
        m[i] = one(T)
    end
    return m
end

function _apply_kcomp(h::AbstractVector, k::Integer, α::Real)
    T = eltype(h)
    pw, pl, nw, nl = @ignore_derivatives _kcomp_indices(h, k)
    winner_mask = @ignore_derivatives _kcomp_mask(length(h), vcat(pw, nw), T)
    pos_winner_mask = @ignore_derivatives _kcomp_mask(length(h), pw, T)
    neg_winner_mask = @ignore_derivatives _kcomp_mask(length(h), nw, T)

    Epos = isempty(pl) ? zero(T) : sum(@view h[pl])
    Eneg = isempty(nl) ? zero(T) : sum(@view h[nl])

    return h .* winner_mask .+ T(α) * Epos .* pos_winner_mask .+ T(α) * Eneg .* neg_winner_mask
end

function (l::KCompetetive)(x::AbstractVector, ps, st::NamedTuple)
    h = ps.weight * x .+ ps.bias
    return l.activation.(_apply_kcomp(h, l.k, l.alpha)), st
end

function (l::KCompetetive)(x::AbstractMatrix, ps, st::NamedTuple)
    h = ps.weight * x .+ ps.bias
    cols = [_apply_kcomp(view(h, :, j), l.k, l.alpha) for j in axes(h, 2)]
    return l.activation.(reduce(hcat, cols)), st
end

# ----------------------------------------------------------------------------
# Text preprocessing utilities (legacy; preserved from original thesis code)
# ----------------------------------------------------------------------------

non_empty_string(s) = s != ""

function count_words(text::AbstractVector{<:AbstractString})
    wc = Dict{String, Int}()
    @inbounds for w in text
        wc[String(w)] = get(wc, String(w), 0) + 1
    end
    return wc
end

function _nl(word::AbstractString, wc::Dict{String, Int}, V::Int)
    n = wc[String(word)]
    logwc = log(1 + n)
    return logwc / V * logwc
end

function normalize_log(text::AbstractVector{<:AbstractString}, wc::Dict{String, Int})
    V = length(wc)
    return [_nl(w, wc, V) for w in text]
end

"""
    transform_text_to_input(src; limit=1000)

Load text from `src` (file path), tokenize on non-word characters
(preserving German umlauts and ampersands), and return `(input, word_counts)`
where `input` is the log-normalized word-count vector of the first `limit` tokens.
"""
function transform_text_to_input(src::AbstractString; limit::Integer = 1000)
    raw = read(src, String)
    toks = String.(filter(non_empty_string, split(raw, r"[^\wäÄöÖüÜ&]+")))
    toks = toks[1:min(limit, length(toks))]
    wc = count_words(toks)
    return normalize_log(toks, wc), wc
end

"""
    get_similar_words(W, query_id, vocab; topn=10)

Given a weight matrix `W` with one row per vocabulary token, return the `topn`
most similar tokens to `vocab[query_id]` under cosine similarity.
"""
function get_similar_words(W::AbstractMatrix, query_id::Integer,
                           vocab::AbstractVector; topn::Integer = 10)
    T = eltype(W)
    row_norms = sqrt.(sum(abs2, W; dims = 2))
    safe_norms = max.(row_norms, T(eps(real(one(T)))))
    Wn = W ./ safe_norms
    scores = vec(Wn * Wn[query_id, :])
    idx = sortperm(scores; rev = true)[1:min(topn, length(scores))]
    return [vocab[i] for i in idx]
end

end # module KATE
