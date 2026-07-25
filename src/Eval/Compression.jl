"""
    Eval.Compression

Compression-side metrics for the plan's "theoretical spine" (MDL /
rate–distortion framing): quantify how well a learned representation
describes its source relative to a general-purpose compressor.

Ship:
- [`dictionary_size`] — number of distinct codes / templates.
- [`bpc_gzip`]        — bits per character after gzip — the
  general-purpose-compressor baseline every learned code should beat.
- [`bpc_dictionary`]  — Shannon entropy of the empirical code
  distribution, expressed per *character of the original corpus*.
  For a good log-template dictionary this should undercut gzip.
- [`codebook_perplexity`] — `exp(H)` of the empirical codebook usage;
  matches the convention in VQ-VAE / BPE papers (a codebook with
  uniform usage has perplexity equal to its size).
- [`entropy_bits`] — helper: Shannon entropy of a label vector in bits.
"""
module Compression

using CodecZlib: GzipCompressor, GzipCompressorStream, transcode
using Statistics

export dictionary_size,
       bpc_gzip, bpc_dictionary,
       codebook_perplexity, entropy_bits

# ---------------------------------------------------------------------------
# Dictionary size
# ---------------------------------------------------------------------------

"Number of distinct codes / templates in a label vector."
dictionary_size(labels::AbstractVector) = length(Set(labels))

# ---------------------------------------------------------------------------
# Entropies
# ---------------------------------------------------------------------------

"""
    entropy_bits(labels::AbstractVector) -> Float64

Shannon entropy (in bits) of the *empirical label distribution* over
`labels`. Two labels are considered equal iff `==` says so, so string
templates, integer codes, and tuples all work. To score a
pre-computed histogram instead, use [`entropy_bits_from_counts`].
"""
function entropy_bits(labels::AbstractVector)
    isempty(labels) && return 0.0
    cnt = Dict{Any, Int}()
    @inbounds for l in labels
        cnt[l] = get(cnt, l, 0) + 1
    end
    return entropy_bits_from_counts(collect(values(cnt)))
end

"""
    entropy_bits_from_counts(counts::AbstractVector{<:Integer}) -> Float64

Shannon entropy (in bits) of a pre-computed histogram. `counts[i]` is
the number of occurrences of the i-th bin; the normalisation is
`sum(counts)`.
"""
function entropy_bits_from_counts(counts::AbstractVector{<:Integer})
    total = sum(counts)
    total == 0 && return 0.0
    h = 0.0
    @inbounds for c in counts
        c == 0 && continue
        p = c / total
        h -= p * log2(p)
    end
    return h
end

export entropy_bits_from_counts

# ---------------------------------------------------------------------------
# BPC — bits per character
# ---------------------------------------------------------------------------

"""
    bpc_gzip(corpus::AbstractString; level = 6) -> Float64
    bpc_gzip(lines::AbstractVector{<:AbstractString}; level = 6) -> Float64

Bits-per-character after gzip at compression `level` (1 fastest,
9 maximal, 6 the stdlib default). Lines are joined with `\\n` before
compression so the result also includes line-separator bytes.

This is the "what does an off-the-shelf compressor achieve?" baseline
from the plan's theoretical spine. Any claim that a learned codebook
is MDL-competitive has to beat it.
"""
function bpc_gzip(corpus::AbstractString; level::Integer = 6)
    isempty(corpus) && return 0.0
    bytes = codeunits(corpus)
    compressed = transcode(GzipCompressor(; level = Int(level)),
                           collect(bytes))
    return 8 * length(compressed) / length(bytes)
end

bpc_gzip(lines::AbstractVector{<:AbstractString}; kwargs...) =
    bpc_gzip(join(lines, '\n'); kwargs...)

"""
    bpc_dictionary(corpus::AbstractString, labels::AbstractVector) -> Float64
    bpc_dictionary(lines::AbstractVector, labels::AbstractVector) -> Float64

Bits per character when we encode each line as its template id under
the empirical distribution of the given labels. Specifically:

    (entropy_bits(labels) * N) / total_characters(corpus)

where `N == length(labels)`. In log terms: "how many bits do we need
per original character if we replace each line with its cluster id".
A dictionary that groups N lines into K templates (K ≪ N) trivially
undercuts gzip *for the id stream*; this metric is meant to be
compared alongside the dictionary size to avoid being gamed by
assigning every line its own template.
"""
function bpc_dictionary(corpus::AbstractString, labels::AbstractVector)
    n_chars = lastindex(corpus) == 0 ? 0 : length(codeunits(corpus))
    n_chars == 0 && return 0.0
    h = entropy_bits(labels)
    return h * length(labels) / n_chars
end

bpc_dictionary(lines::AbstractVector{<:AbstractString}, labels::AbstractVector) =
    bpc_dictionary(join(lines, '\n'), labels)

# ---------------------------------------------------------------------------
# Codebook perplexity
# ---------------------------------------------------------------------------

"""
    codebook_perplexity(labels::AbstractVector) -> Float64

`exp(H_nats(labels))` where `H_nats` is the natural-base entropy of the
empirical distribution over labels. Equals the dictionary size iff
every code is used exactly equally; smaller than the dictionary size
when usage is concentrated. Matches the convention used by the VQ-VAE
and BPE literatures.
"""
function codebook_perplexity(labels::AbstractVector)
    isempty(labels) && return 0.0
    cnt = Dict{Any, Int}()
    @inbounds for l in labels
        cnt[l] = get(cnt, l, 0) + 1
    end
    total = length(labels)
    h = 0.0
    @inbounds for c in values(cnt)
        p = c / total
        h -= p * log(p)
    end
    return exp(h)
end

end # module Compression
