using Test
using LogClustering
using LogClustering.ADWINs: ADWIN, last_drop_mean
# `update!` / `width` are also exported by other modules (Instance's
# ValueNoveltyDetector, …); importing them as barewords would shadow
# those for later test files sharing this Main. Alias to keep runtests
# order-independent.
using LogClustering.ADWINs: update! as adwin_update!, width as adwin_width
using Statistics: mean
using Random: MersenneTwister

@testset "Stats.ADWIN" begin

    @testset "constructor validation" begin
        @test_throws ArgumentError ADWIN(; delta = 0.0)
        @test_throws ArgumentError ADWIN(; delta = 1.0)
        a = ADWIN()
        @test adwin_width(a) == 0
        @test isnan(mean(a))
    end

    @testset "stable stream: window stays bounded, mean tracks" begin
        a = ADWIN(; delta = 0.002)
        rng = MersenneTwister(0)
        for _ in 1:2000
            adwin_update!(a, 10.0 + randn(rng))
        end
        @test length(a.buckets) < 40           # O(log n) buckets, bounded
        @test abs(mean(a) - 10.0) < 2.0        # adaptive mean ≈ the level
    end

    @testset "detects an upward step change and adapts the mean" begin
        a = ADWIN(; delta = 0.002)
        rng = MersenneTwister(1)
        for _ in 1:500; adwin_update!(a, 10.0 + randn(rng)); end
        mean_before = mean(a)
        @test abs(mean_before - 10.0) < 2.0
        # Capture the direction AT the first detection: last_drop_mean
        # reflects the latest drop, which in steady state becomes the
        # new regime — so it's only a change-direction signal at the
        # moment the change fires (which is how the rule engine reads it).
        drop_at_change = NaN
        for _ in 1:300
            if adwin_update!(a, 50.0 + randn(rng)) && isnan(drop_at_change)
                drop_at_change = last_drop_mean(a)
            end
        end
        @test !isnan(drop_at_change)            # the 10→50 jump is caught
        @test drop_at_change < 20.0             # it dropped the old (low) window
        @test mean(a) > 40.0                    # window adapted to the new regime
    end

    @testset "detects a downward change (direction via last_drop_mean)" begin
        a = ADWIN(; delta = 0.002)
        rng = MersenneTwister(2)
        for _ in 1:500; adwin_update!(a, 50.0 + randn(rng)); end
        drop_at_change = NaN
        for _ in 1:300
            if adwin_update!(a, 10.0 + randn(rng)) && isnan(drop_at_change)
                drop_at_change = last_drop_mean(a)
            end
        end
        @test !isnan(drop_at_change)
        @test drop_at_change > 40.0             # dropped the HIGH regime at the shift
        @test mean(a) < 20.0
    end

    @testset "constant stream never false-alarms" begin
        a = ADWIN(; delta = 0.002)
        fired = 0
        for _ in 1:1000
            adwin_update!(a, 7.0) && (fired += 1)
        end
        @test fired == 0                        # no change in a constant series
        @test mean(a) ≈ 7.0
    end

end
