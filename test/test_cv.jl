using Test
using Random
using LogClustering
using LogClustering.CV
using LogClustering.CV: Split, time_ordered_split, time_ordered_kfold,
                       per_host_split, per_host_kfold, stratified_by

@testset "Eval.CV" begin
    @testset "Split — disjointness invariant" begin
        s = Split(1:3, 4:5)
        @test s.train == [1, 2, 3]
        @test s.test  == [4, 5]
        @test length(s) == 5
        @test_throws ArgumentError Split([1, 2, 3], [3, 4])   # overlap
    end

    @testset "time_ordered_split — prefix / suffix" begin
        s = time_ordered_split(10; train_frac = 0.7)
        @test s.train == collect(1:7)
        @test s.test  == collect(8:10)
        @test minimum(s.test) > maximum(s.train)         # temporal order
        # Edge cases
        @test_throws ArgumentError time_ordered_split(1)
        @test_throws ArgumentError time_ordered_split(10; train_frac = 0)
        @test_throws ArgumentError time_ordered_split(10; train_frac = 1)
    end

    @testset "time_ordered_kfold — walk-forward monotonic train size" begin
        folds = time_ordered_kfold(10, 5)
        @test length(folds) == 4
        # Train sizes grow strictly, test sets are disjoint and contiguous.
        sizes = [length(f.train) for f in folds]
        @test all(diff(sizes) .> 0)
        # All test indices together partition 3..10 (fold 1 skipped when
        # min_train = 1 so its train would be empty).
        tests = sort!(reduce(vcat, (f.test for f in folds)))
        @test tests == collect(3:10)
        for f in folds
            @test minimum(f.test) > maximum(f.train)     # temporal order holds
        end
    end

    @testset "per_host_split — leave one, leave many" begin
        hosts = ["a", "a", "b", "b", "c", "c", "c"]
        s = per_host_split(hosts; held_out = "c")
        @test s.train == [1, 2, 3, 4]
        @test s.test  == [5, 6, 7]

        s = per_host_split(hosts; held_out = ["b", "c"])
        @test s.train == [1, 2]
        @test s.test  == [3, 4, 5, 6, 7]

        @test_throws ArgumentError per_host_split(hosts; held_out = "x")   # no test
        @test_throws ArgumentError per_host_split(hosts;
            held_out = ["a", "b", "c"])                                    # no train
    end

    @testset "per_host_kfold — every host appears in exactly one test fold" begin
        hosts = ["h$(mod(i, 10))" for i in 1:200]
        rng = Random.MersenneTwister(0)
        folds = per_host_kfold(hosts; k = 5, rng = rng)
        @test length(folds) == 5
        # No host should appear in both train and test of the same fold.
        for f in folds
            train_hosts = Set(hosts[i] for i in f.train)
            test_hosts  = Set(hosts[i] for i in f.test)
            @test isempty(intersect(train_hosts, test_hosts))
        end
        # Every unique host appears in some fold's test set.
        unioned = Set{String}()
        for f in folds
            union!(unioned, Set(hosts[i] for i in f.test))
        end
        @test unioned == Set(hosts)
    end

    @testset "stratified_by — preserves per-label proportions" begin
        labels = repeat(["a", "b", "c"], inner = 20)    # 20 of each
        rng = Random.MersenneTwister(1)
        s = stratified_by(labels; train_frac = 0.75, rng = rng)
        # 75 % of 20 = 15 per class → train has 15 each, test has 5 each.
        counts_train = Dict(l => count(==(l), labels[s.train]) for l in unique(labels))
        counts_test  = Dict(l => count(==(l), labels[s.test])  for l in unique(labels))
        for l in unique(labels)
            @test counts_train[l] == 15
            @test counts_test[l]  == 5
        end
    end
end
