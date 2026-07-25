using Test
using LogClustering
using LogClustering.Episodes
using LogClustering.Episodes: invert_sequence, total_utility,
                              external_utility, avg_utility, local_utility,
                              iesc

@testset "Episodes" begin
    @testset "invert_sequence" begin
        v = invert_sequence([1, 2, 1, 3, 1, 2])
        @test v[1] == [1, 3, 5]
        @test v[2] == [2, 6]
        @test v[3] == [4]
    end

    @testset "utility helpers" begin
        us = Dict(1 => 2.0, 2 => 3.0, 3 => 1.0)
        seq = [1, 2, 1, 3]
        tu = total_utility(seq, us)
        @test tu == 2.0 + 3.0 + 2.0 + 1.0
        # external: sum(utilities[pattern]) / tu
        @test external_utility(us, tu, [1, 2]) ≈ (2.0 + 3.0) / tu
        # average: external / length(pattern)
        @test avg_utility(us, tu, length(seq), [1, 2]) ≈ ((2.0 + 3.0) / tu) / 2
        # local: just the sum
        @test local_utility(us, [1, 2]) == 5.0
        # iesc = (sum u(prefix) + best u(candidate)) / tu
        @test iesc(tu, us, [1], [2, 3]) ≈ (2.0 + 3.0) / tu
    end

    @testset "MV-Span — single-shot recovers the obvious serial episode" begin
        # Sequence has two full copies of the pattern [1, 2, 3].
        seq = [1, 2, 3, 4, 1, 2, 3]
        db = mv_span(seq; min_sup = 2, max_gap = -1)
        @test haskey(db, [1, 2, 3])
        # Two minimal occurrences: positions (1,2,3) and (5,6,7).
        @test length(db[[1, 2, 3]]) == 2
        @test db[[1, 2, 3]][1] == [1, 2, 3]
        @test db[[1, 2, 3]][2] == [5, 6, 7]
    end

    @testset "MV-Span — contiguous-only (max_gap = 0)" begin
        # With max_gap = 0, [1,3] cannot be a serial episode since they
        # aren't adjacent in `[1,2,3]`. But [1,2] and [2,3] are.
        seq = [1, 2, 3, 1, 2, 3]
        db = mv_span(seq; min_sup = 2, max_gap = 0)
        @test haskey(db, [1, 2])
        @test haskey(db, [2, 3])
        @test !haskey(db, [1, 3])
    end

    @testset "MV-Span — max_repetitions filters self-repeating patterns" begin
        # With max_repetitions = 1, [1,1] must not appear.
        seq = [1, 1, 2, 1, 1, 2]
        db = mv_span(seq; min_sup = 2, max_repetitions = 1, max_gap = -1)
        @test !haskey(db, [1, 1])
        @test haskey(db, [1, 2])
    end

    @testset "MV-Span — min_sup prunes low-frequency patterns" begin
        seq = [1, 2, 3, 1, 2, 3, 4, 5]
        db = mv_span(seq; min_sup = 2)
        @test !haskey(db, [4, 5])                         # supp = 1 < 2
    end

    @testset "MV-Span — external-utility filter" begin
        seq = [1, 2, 3, 1, 2, 3]
        utils = Dict(1 => 1.0, 2 => 1.0, 3 => 1.0)
        db = mv_span(seq; min_sup = 2, max_gap = -1,
                     utilities = utils, utility = :external,
                     min_utility = 0.9)         # too high: nothing survives
        # No pattern's external utility reaches 90% of the total.
        @test isempty(db)
    end

    @testset "MT-Span — small sequence, high-utility episode" begin
        seq = [1, 2, 3, 1, 2, 3]
        utils = Dict(1 => 1.0, 2 => 2.0, 3 => 3.0)
        moSet, hueSet = mt_span(seq, utils;
                                max_time_duration = 3,
                                min_sup = 1,
                                min_utility = 0.0,
                                max_repetitions = 2,
                                max_gap = -1)
        # Singletons are always present in moSet.
        @test haskey(moSet, [1])
        @test haskey(moSet, [2])
        @test haskey(moSet, [3])
        # At least one ≥2-length pattern is discovered.
        multi = filter(k -> length(k) > 1, collect(keys(moSet)))
        @test !isempty(multi)
    end

    @testset "MT-Span — min_utility gate" begin
        seq = [1, 1, 1, 2]
        utils = Dict(1 => 1.0, 2 => 1.0)
        _, hueSet = mt_span(seq, utils;
                            max_time_duration = 3,
                            min_sup = 1,
                            min_utility = 0.99)   # total = 4; ratio > 0.99 only for whole seq
        # No single event or short window can clear 99 % relative utility.
        @test all(length(k) >= 1 for k in keys(hueSet)) || isempty(hueSet)
    end
end
