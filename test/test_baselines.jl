using Test
using Random
using LogClustering
using LogClustering.Baselines
using LogClustering.Baselines: prefixspan, spade, cmspade

@testset "Mining.Baselines" begin
    @testset "all three miners agree on a small fixture" begin
        seq = [1, 2, 3, 1, 2, 3, 4, 1, 2]
        for cfg in (
            (min_sup = 2, max_gap = -1, max_time_duration = -1),
            (min_sup = 2, max_gap = 1,  max_time_duration = -1),
            (min_sup = 1, max_gap = -1, max_time_duration = 3),
            (min_sup = 3, max_gap = -1, max_time_duration = -1),
        )
            ps = prefixspan(seq; cfg...)
            sp = spade(seq; cfg...)
            cm = cmspade(seq; cfg...)
            @test sort(collect(keys(ps))) == sort(collect(keys(sp)))
            @test sort(collect(keys(sp))) == sort(collect(keys(cm)))
            for k in keys(ps)
                @test length(ps[k]) == length(sp[k])
                @test length(sp[k]) == length(cm[k])
            end
        end
    end

    @testset "singletons are filtered (matches mv_span's default)" begin
        seq = [1, 1, 2, 2, 3, 3]
        for fn in (prefixspan, spade, cmspade)
            d = fn(seq; min_sup = 2)
            @test all(length(k) >= 2 for k in keys(d))
        end
    end

    @testset "min_sup gate" begin
        seq = [1, 2, 1, 2, 1, 2]   # the only multi-pattern is [1,2] with 3 occurrences
        for fn in (prefixspan, spade, cmspade)
            @test haskey(fn(seq; min_sup = 2), [1, 2])
            @test !haskey(fn(seq; min_sup = 4), [1, 2])
        end
    end

    @testset "max_gap drops far-apart occurrences" begin
        # `[1, 2]` occurs at positions (1,2), (5,6), (9,10) — every
        # pair is one position apart, so max_gap=1 keeps them all.
        # `[1, 3]` occurs at (1, 7), (5, 11) — gap 6 and 6, dropped at gap=1.
        seq = [1, 2, 0, 0, 1, 2, 3, 0, 1, 2, 3]
        for fn in (prefixspan, spade, cmspade)
            @test haskey(fn(seq; min_sup = 2, max_gap = 1), [1, 2])
            @test !haskey(fn(seq; min_sup = 2, max_gap = 1), [1, 3])
            @test haskey(fn(seq; min_sup = 2, max_gap = -1), [1, 3])
        end
    end

    @testset "max_time_duration caps pattern length" begin
        seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        for fn in (prefixspan, spade, cmspade)
            d = fn(seq; min_sup = 2, max_time_duration = 2)
            @test all(length(k) <= 3 for k in keys(d))
        end
    end

    @testset "occurrence positions are valid 1-based indices into the sequence" begin
        seq = [1, 2, 3, 1, 2, 3, 4]
        for fn in (prefixspan, spade, cmspade)
            d = fn(seq; min_sup = 2)
            for (pat, occs) in d
                for occ in occs
                    @test length(occ) == length(pat)
                    @test all(1 .<= occ .<= length(seq))
                    # Each position holds the right event.
                    @test [seq[i] for i in occ] == pat
                    # Strictly increasing.
                    @test issorted(occ; lt = <)
                end
                # Non-overlapping: each occurrence starts after the
                # previous one ends.
                for i in 2:length(occs)
                    @test occs[i][1] > occs[i - 1][end]
                end
            end
        end
    end

    @testset "empty / single-element sequence yields nothing" begin
        for fn in (prefixspan, spade, cmspade)
            @test isempty(fn(Int[]; min_sup = 1))
            @test isempty(fn([7]; min_sup = 1))   # only a singleton
        end
    end

    @testset "scales to a 200-event sequence" begin
        rng = Random.MersenneTwister(0)
        seq = rand(rng, 1:5, 200)
        for fn in (prefixspan, spade, cmspade)
            d = fn(seq; min_sup = 5, max_time_duration = 4)
            @test !isempty(d)
            @test all(2 <= length(k) <= 5 for k in keys(d))
            @test all(length(v) >= 5 for v in values(d))
        end
    end
end
