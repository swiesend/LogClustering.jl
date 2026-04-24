using Test
using LogClustering
using LogClustering.Featurise
using LogClustering.Featurise: Vocabulary, build_vocab, tokenise,
                               tokenise_ids, bow, sequence_matrix

const TOY = [
    "INFO started pid 1001",
    "INFO started pid 1002",
    "ERROR from 10.0.0.1",
    "ERROR from 10.0.0.2",
    "INFO ready",
]

@testset "Featurise" begin
    @testset "tokenise — whitespace split without separators" begin
        @test tokenise("a  b c") == ["a", "b", "c"]
        @test tokenise("  ")    == SubString{String}[]
    end

    @testset "build_vocab — masking collapses typed slots" begin
        v = build_vocab(TOY; mask = true)
        @test v.mask
        @test "<UNK>" in v.tokens
        @test v.tokens[1] == "<UNK>"
        # `<IP>` and `<INT>` are single vocab slots despite four
        # distinct raw IPs / pids in the input.
        @test "<IP>" in v.tokens
        @test "<INT>" in v.tokens
        @test length(v) == length(unique(v.tokens))
    end

    @testset "build_vocab — min_count + max_vocab cap the output" begin
        v = build_vocab(TOY; min_count = 2)
        # `ready` appears once → dropped under min_count=2.
        @test !("ready" in v.tokens)

        v2 = build_vocab(TOY; max_vocab = 3)
        @test length(v2) == 3   # <UNK> + top-2 by frequency
    end

    @testset "tokenise_ids — unknown tokens map to 1" begin
        v = build_vocab(["a b c"])
        ids = tokenise_ids("a b z", v)
        @test length(ids) == 3
        @test ids[1] != 1              # known
        @test ids[3] == 1              # <UNK>
    end

    @testset "bow — shape + non-negative counts" begin
        v = build_vocab(TOY)
        X = bow(TOY, v; normalise = :count)
        @test size(X) == (length(v), length(TOY))
        @test all(X .>= 0)
        @test any(X .> 0)
    end

    @testset "bow — :l1 columns sum to 1 (when non-empty)" begin
        v = build_vocab(TOY)
        X = bow(TOY, v; normalise = :l1)
        for j in axes(X, 2)
            s = sum(X[:, j])
            @test isapprox(s, 1; atol = 1e-6)
        end
    end

    @testset "bow — :binary yields values in {0, 1}" begin
        v = build_vocab(TOY)
        X = bow(TOY, v; normalise = :binary)
        @test all(x -> x == 0 || x == 1, X)
    end

    @testset "bow — :log stays finite and non-negative" begin
        v = build_vocab(TOY)
        X = bow(TOY, v; normalise = :log)
        @test all(isfinite, X)
        @test all(X .>= 0)
    end

    @testset "bow — rejects unknown normalisations" begin
        v = build_vocab(TOY)
        @test_throws ArgumentError bow(TOY, v; normalise = :not_a_thing)
    end

    @testset "sequence_matrix — shape + right-alignment + padding" begin
        v = build_vocab(TOY)
        S = sequence_matrix(TOY, v; seqlen = 6)
        @test size(S) == (6, length(TOY))
        # Lines shorter than seqlen are left-padded with <UNK>=1.
        for j in axes(S, 2)
            ids = tokenise_ids(TOY[j], v)
            k = min(length(ids), 6)
            @test S[6 - k + 1:end, j] == ids[1:k]
            if 6 > k
                @test all(S[1:6 - k, j] .== 1)
            end
        end
    end

    @testset "sequence_matrix — rejects silly seqlen" begin
        v = build_vocab(TOY)
        @test_throws ArgumentError sequence_matrix(TOY, v; seqlen = 0)
    end

    # -----------------------------------------------------------------
    # Inference-time OOV policies (:nearest, :distribute)
    # -----------------------------------------------------------------

    @testset "build_ngram_profile — shape + sentinels" begin
        v = build_vocab(["error login", "warning logout"]; mask = false)
        p = Featurise.build_ngram_profile(v; n = 3)
        @test p.n == 3
        @test length(p.grams) == length(v)
        # id 1 is UNK → empty profile.
        @test isempty(p.grams[1])
        # "login" → ["<lo", "log", "ogi", "gin", "in>"]
        idx = v.index["login"]
        @test "<lo" in p.grams[idx]
        @test "in>" in p.grams[idx]
        # n must be ≥ 1.
        @test_throws ArgumentError Featurise.build_ngram_profile(v; n = 0)
    end

    @testset "tokenise_ids — :nearest routes OOV to closest by Jaccard" begin
        v = build_vocab(["error login", "warning logout"]; mask = false)
        p = Featurise.build_ngram_profile(v)
        # Known token unchanged.
        @test tokenise_ids("login", v; oov_policy = :nearest, profile = p) ==
              [v.index["login"]]
        # OOV "loginn" shares many 3-grams with "login" → routed there.
        ids = tokenise_ids("loginn", v; oov_policy = :nearest,
                           profile = p, oov_min_sim = 0.3)
        @test ids == [v.index["login"]]
        # OOV totally unlike any vocab token → UNK.
        ids_far = tokenise_ids("zzzzzzz", v; oov_policy = :nearest,
                               profile = p, oov_min_sim = 0.3)
        @test ids_far == [1]
    end

    @testset "tokenise_ids — :distribute rejected at the integer entry" begin
        v = build_vocab(["a b c"]; mask = false)
        @test_throws ArgumentError tokenise_ids("x", v;
            oov_policy = :distribute)
    end

    @testset "tokenise_ids — unknown oov_policy is rejected" begin
        v = build_vocab(["a b"]; mask = false)
        @test_throws ArgumentError tokenise_ids("x", v; oov_policy = :bogus)
    end

    @testset "bow — :unk is the backward-compat default" begin
        v = build_vocab(["error login", "warning logout"]; mask = false)
        X_default  = bow(["totally new"], v; normalise = :count)
        X_explicit = bow(["totally new"], v; normalise = :count,
                         oov_policy = :unk)
        @test X_default == X_explicit
    end

    @testset "bow — :nearest routes OOV to closest known token" begin
        v = build_vocab(["error login", "warning logout"]; mask = false)
        X = bow(["loginn"], v; normalise = :count, oov_policy = :nearest,
                oov_min_sim = 0.3)
        @test X[v.index["login"], 1] ≈ 1.0
        # Below threshold → UNK.
        X_far = bow(["qqqqqq"], v; normalise = :count, oov_policy = :nearest,
                    oov_min_sim = 0.5)
        @test X_far[1, 1] ≈ 1.0
    end

    @testset "bow — :distribute produces fractional top-k routing" begin
        v = build_vocab(["login logout"]; mask = false)
        X = bow(["logisn"], v; normalise = :count,
                oov_policy = :distribute, oov_top_k = 2, oov_min_sim = 0.05)
        # Exactly one OOV token → column sums to 1 (weighted split across
        # top-k similar vocab tokens).
        @test sum(X[:, 1]) ≈ 1.0 atol = 1e-6
        # Mass lands on login + logout, not UNK.
        @test X[1, 1] == 0
        @test X[v.index["login"],  1] > 0
        @test X[v.index["logout"], 1] > 0
    end

    @testset "bow — :distribute falls through to UNK when no match clears sim threshold" begin
        v = build_vocab(["alpha beta"]; mask = false)
        X = bow(["zz"], v; normalise = :count,
                oov_policy = :distribute, oov_min_sim = 0.99)
        @test X[1, 1] ≈ 1.0
    end

    @testset "sequence_matrix — :nearest threads through; :distribute rejected" begin
        v = build_vocab(["error login", "warning logout"]; mask = false)
        S = sequence_matrix(["loginn"], v; seqlen = 4,
                            oov_policy = :nearest, oov_min_sim = 0.3)
        @test S[end, 1] == v.index["login"]         # right-aligned
        @test_throws ArgumentError sequence_matrix(["x"], v; seqlen = 4,
            oov_policy = :distribute)
    end

    @testset "bow — rejects unknown oov_policy" begin
        v = build_vocab(["a b"]; mask = false)
        @test_throws ArgumentError bow(["x"], v; oov_policy = :bogus)
    end
end
