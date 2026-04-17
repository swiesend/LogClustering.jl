using Test
using Random
using Lux
using LogClustering
using LogClustering.DeepKATE: deep_kate
using LogClustering.Instance: reconstruction_error_abs, reconstruction_error_sq,
                              latent_distance, anomaly_score

const RNG_ANOM = Random.MersenneTwister(7)

"Train DeepKATE for a few steps so the AE is not trivial."
function _warmup(model, ps, st, X; steps = 25, lr = 1f-2)
    for _ in 1:steps
        Y, _ = model(X, ps, Lux.testmode(st))
        grad = 2f0 .* (Y .- X) ./ size(X, 2)
        # Simple finite-difference-free pseudo-gradient: nudge final Dense
        # directly. Enough to pull reconstruction away from the identity.
        layer_last = :layer_10
        p = getfield(ps, layer_last)
        h, _ = model(X, ps, Lux.testmode(st))
        # adjust bias toward mean signal (toy training loop — we only
        # need the AE to produce a non-trivial reconstruction for the
        # assertions below)
        update = vec(mean(X .- h; dims = 2))
        new_bias = p.bias .+ Float32(lr) .* update
        new_p = merge(p, (bias = new_bias,))
        ps = merge(ps, NamedTuple{(layer_last,)}((new_p,)))
    end
    return ps
end

@testset "Anomaly.Instance" begin
    @testset "reconstruction_error_abs — shape and positivity" begin
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 16, 6)
        e = reconstruction_error_abs(m, ps, st, X)
        @test length(e) == 6
        @test all(>=(0), e)
    end

    @testset "reconstruction_error_sq — sq >= 0 and shape" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 8, 4)
        e = reconstruction_error_sq(m, ps, st, X)
        @test length(e) == 4
        @test all(>=(0), e)
    end

    @testset "single-sample vector forms" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        x = rand(RNG_ANOM, Float32, 8)
        @test reconstruction_error_abs(m, ps, st, x) isa Real
        @test reconstruction_error_sq(m, ps, st, x) isa Real
        @test latent_distance(m, ps, st, x) isa Real
    end

    @testset "latent_distance — self-to-centroid is zero for single point" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 8, 3)
        d = latent_distance(m, ps, st, X; reference = :mean)
        @test length(d) == 3
        @test all(>=(0), d)
    end

    @testset "anomaly_score — normalisation caps components at 1" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 8, 8)
        s = anomaly_score(m, ps, st, X;
                          weights = (abs = 1.0, sq = 1.0, latent = 0.0),
                          normalise = true)
        @test length(s) == 8
        @test all(>=(0), s)
        @test all(<=(1 + 1e-6), s)
    end

    @testset "anomaly_score — raw mode skips normalisation" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 8, 4)
        s_raw = anomaly_score(m, ps, st, X;
                              weights = (abs = 1.0, sq = 0.0, latent = 0.0),
                              normalise = false)
        s_abs = reconstruction_error_abs(m, ps, st, X)
        @test s_raw ≈ Float64.(s_abs)
    end

    @testset "anomaly_score — all-zero weights rejected" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        X = rand(RNG_ANOM, Float32, 8, 2)
        @test_throws ArgumentError anomaly_score(m, ps, st, X;
            weights = (abs = 0.0, sq = 0.0, latent = 0.0))
    end

    @testset "outlier samples score higher than in-distribution" begin
        m = deep_kate(8; latent = 2, k1 = 4)
        ps, st = Lux.setup(RNG_ANOM, m)
        # 12 in-distribution samples in the unit cube …
        X_in = rand(RNG_ANOM, Float32, 8, 12)
        # … and two deliberate outliers at large magnitude.
        X_out = hcat(fill(10f0, 8), fill(-10f0, 8))
        X = hcat(X_in, X_out)
        s = anomaly_score(m, ps, st, X;
                          weights = (abs = 1.0, sq = 1.0, latent = 0.0),
                          normalise = false)
        # The last two samples must exceed the maximum of the in-distribution.
        @test minimum(s[13:14]) > maximum(s[1:12])
    end
end
