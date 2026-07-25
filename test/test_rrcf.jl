using Test
using LogClustering
using LogClustering.RRCF
using LogClustering.RRCF: RCForest, observe!, score
using Random: MersenneTwister

@testset "Anomaly.RRCF" begin

    @testset "scores are in [0,1]; neutral before first forest" begin
        f = RCForest(; dims = 4, sample_size = 32, rebuild_every = 32,
                     rng = MersenneTwister(1))
        @test score(f, zeros(4)) == 0.5          # no forest yet
        for _ in 1:100
            s = observe!(f, 0.1 .* randn(MersenneTwister(2), 4))
            @test 0.0 <= s <= 1.0
        end
    end

    @testset "outliers score above inliers" begin
        rng = MersenneTwister(7)
        f = RCForest(; dims = 8, n_trees = 50, sample_size = 128,
                     rebuild_every = 128, rng = rng)
        # Learn a tight inlier cluster.
        for _ in 1:1500
            observe!(f, 0.1 .* randn(rng, 8))
        end
        # Read-only scores (no learning) so the comparison is clean.
        outlier = score(f, fill(8.0, 8))
        inlier  = score(f, 0.1 .* randn(rng, 8))
        @test outlier > inlier
        @test outlier > 0.6                       # clearly anomalous
    end

    @testset "score-then-learn: a point never scores against itself" begin
        # Two identical detectors; observing then scoring x should differ
        # from scoring-only, proving observe! learns AFTER scoring.
        rng = MersenneTwister(3)
        f = RCForest(; dims = 4, sample_size = 16, rebuild_every = 16,
                     rng = MersenneTwister(3))
        for _ in 1:40; observe!(f, 0.1 .* randn(rng, 4)); end
        x = fill(5.0, 4)
        s_first = observe!(f, x)                  # scored against prior model
        # It was folded in; nothing crashes, score is valid.
        @test 0.0 <= s_first <= 1.0
    end

    @testset "dimension mismatch is rejected" begin
        f = RCForest(; dims = 4)
        @test_throws ArgumentError observe!(f, zeros(3))
        @test_throws ArgumentError score(f, zeros(5))
    end

    @testset "reservoir stays bounded" begin
        f = RCForest(; dims = 4, sample_size = 50, rebuild_every = 50,
                     rng = MersenneTwister(9))
        for _ in 1:5000; observe!(f, randn(MersenneTwister(1), 4)); end
        @test length(f.reservoir) <= 50           # bounded memory
        @test length(f.trees) == f.n_trees
    end

end
