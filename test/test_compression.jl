using Test
using LogClustering
using LogClustering.Compression
using LogClustering.Compression: entropy_bits_from_counts

@testset "Eval.Compression" begin
    @testset "dictionary_size — distinct labels" begin
        @test dictionary_size(["a", "a", "b", "c", "a"]) == 3
        @test dictionary_size([1, 1, 1]) == 1
        @test dictionary_size(Int[]) == 0
    end

    @testset "entropy_bits — counts the empirical label distribution" begin
        # Four distinct labels, each appearing once → uniform → log2(4).
        @test entropy_bits(["a", "b", "c", "d"]) ≈ 2.0
        # All-same labels → fully concentrated → 0 bits.
        @test entropy_bits(fill("x", 10)) == 0.0
        # String-label form agrees with the explicit histogram form.
        @test entropy_bits(vcat(fill("a", 3), fill("b", 1))) ≈
              entropy_bits_from_counts([3, 1])
        # Empty input → 0 by convention.
        @test entropy_bits(String[]) == 0.0
    end

    @testset "entropy_bits_from_counts — explicit histogram" begin
        # Uniform 4-bin histogram → 2 bits.
        @test entropy_bits_from_counts([1, 1, 1, 1]) ≈ 2.0
        # Single-bin histogram → 0 bits regardless of magnitude.
        @test entropy_bits_from_counts([10, 0, 0, 0]) == 0.0
        @test entropy_bits_from_counts(Int[]) == 0.0
    end

    @testset "bpc_gzip — redundant corpora compress well" begin
        # Highly redundant corpus: gzip should beat 1 bit per char.
        redundant = repeat("abcabcabcabc ", 50)
        @test 0 < bpc_gzip(redundant) < 1.0
        # Vector form matches the joined-string form.
        lines = ["hello world" for _ in 1:10]
        @test bpc_gzip(lines) ≈ bpc_gzip(join(lines, '\n'))
        # Empty input → 0.
        @test bpc_gzip("") == 0.0
    end

    @testset "bpc_dictionary — linear in label entropy" begin
        lines = [repeat("foo bar", 5) for _ in 1:8]    # 35 chars × 8 + 7 newlines
        # All-same labels → 0 entropy → 0 bpc.
        @test bpc_dictionary(lines, fill("X", 8)) == 0.0
        # Two labels 4-4 → 1 bit × 8 samples / total_chars → small but positive.
        bpc = bpc_dictionary(lines, vcat(fill("X", 4), fill("Y", 4)))
        @test bpc > 0
        @test bpc < 0.1
    end

    @testset "codebook_perplexity — equals dictionary size for uniform usage" begin
        # Uniform over K codes: perplexity = K (to within FP).
        labels = repeat(["a", "b", "c", "d"], inner = 5)
        @test codebook_perplexity(labels) ≈ 4.0
        # Fully concentrated: perplexity = 1.
        @test codebook_perplexity(fill("X", 20)) ≈ 1.0
        # Empty input → 0.
        @test codebook_perplexity([]) == 0.0
    end

    @testset "cross-check: single-template corpus" begin
        # One perfectly-regular template: the dictionary encoding with a
        # single label is 0 bpc; gzip is strictly greater.
        lines = ["INFO service ok" for _ in 1:20]
        gz   = bpc_gzip(lines)
        dict = bpc_dictionary(lines, fill("E1", 20))
        @test dict == 0.0
        @test dict <= gz
    end
end
