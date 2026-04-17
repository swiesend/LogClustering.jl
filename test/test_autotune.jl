using Test
using Random
using Lux
using LogClustering
using LogClustering.AutoTune
using LogClustering.AutoTune: fit_hyperparams, pca_elbow
using LogClustering.DeepKATE: deep_kate
using LogClustering.VQVAE: vq_vae
using LogClustering.SeqLSTM: seq_lstm

const RNG_AT = Random.MersenneTwister(13)

@testset "AutoTune" begin
    @testset "pca_elbow" begin
        # Two-dim-effective data embedded in 10-D; elbow should be 2.
        N = 50
        signal = randn(RNG_AT, Float32, 2, N)
        lift = randn(RNG_AT, Float32, 10, 2)
        X = lift * signal .+ 0.01f0 .* randn(RNG_AT, Float32, 10, N)
        @test pca_elbow(X; var_threshold = 0.95) == 2
        # High threshold pushes the elbow up; clamp still holds.
        @test pca_elbow(X; var_threshold = 0.9999, upper = 5) <= 5
        # Empty dataset falls back to `lower`.
        @test pca_elbow(randn(5, 0); lower = 3) == 3
    end

    @testset "deep_kate heuristic returns a factory-ready NamedTuple" begin
        X = randn(RNG_AT, Float32, 32, 100)
        cfg = fit_hyperparams(:deep_kate, X)
        @test cfg.n == 32
        # Thesis DeepKATE bottleneck is 5 → latent → 5, so `latent` is
        # capped at 5 and `k1` is capped at 100 (the first hidden).
        @test 2 <= cfg.latent <= 5
        @test 4 <= cfg.k1 <= 100
        @test cfg.p isa AbstractFloat
        # Splats cleanly into the factory.
        m = deep_kate(cfg.n; latent = cfg.latent, k1 = cfg.k1, p = cfg.p)
        @test m isa Chain
    end

    @testset "vq_vae heuristic + factory round-trip" begin
        X = randn(RNG_AT, Float32, 24, 80)
        cfg = fit_hyperparams(:vq_vae, X)
        @test cfg.n == 24
        @test 16 <= cfg.codebook_size <= 256
        @test 4 <= cfg.embed_dim <= 64
        @test 32 <= cfg.hidden <= 512
        m = vq_vae(cfg.n; codebook_size = cfg.codebook_size,
                   embed_dim = cfg.embed_dim, hidden = cfg.hidden)
        @test m isa Chain
    end

    @testset "seq_lstm heuristic — matrix and NamedTuple corpus" begin
        seq = [rand(RNG_AT, 1:20) for _ in 1:8, _ in 1:4]
        cfg = fit_hyperparams(:seq_lstm, seq)
        @test cfg.vocab_size >= 1
        @test 8 <= cfg.embed <= 64
        @test 16 <= cfg.hidden <= 256
        m = seq_lstm(cfg.vocab_size; embed = cfg.embed, hidden = cfg.hidden)
        @test m isa Chain

        # Shape-only hint.
        cfg2 = fit_hyperparams(:seq_lstm, (vocab_size = 50, seq_len = 16, batch = 4))
        @test cfg2.vocab_size == 50
    end

    @testset "user overrides survive the merge" begin
        X = randn(RNG_AT, Float32, 16, 40)
        cfg = fit_hyperparams(:deep_kate, X; latent = 3, p = 0.2f0)
        @test cfg.latent == 3
        @test cfg.p == 0.2f0
    end

    @testset "unknown kind errors" begin
        @test_throws ErrorException fit_hyperparams(:not_a_kind, randn(4, 4))
    end

    @testset "budget > 0 runs random search and stays factory-ready" begin
        X = randn(RNG_AT, Float32, 20, 60)
        cfg = fit_hyperparams(:deep_kate, X; budget = 5, rng = RNG_AT)
        @test cfg.n == 20
        @test 2 <= cfg.latent <= 16
        m = deep_kate(cfg.n; latent = cfg.latent, k1 = cfg.k1, p = cfg.p)
        @test m isa Chain
    end

    @testset "budget search on vq_vae perturbs embed_dim around the seed" begin
        X = randn(RNG_AT, Float32, 16, 40)
        base = fit_hyperparams(:vq_vae, X)
        out  = fit_hyperparams(:vq_vae, X; budget = 10, rng = RNG_AT)
        # Search can only improve or match the heuristic (it's a
        # strict argmin over candidates that includes the seed).
        @test 4 <= out.embed_dim <= 64
        @test 32 <= out.hidden <= 512
        @test 16 <= out.codebook_size <= 256
    end
end
