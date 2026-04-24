using Test
using LogClustering
using LogClustering.Metrics
using LogClustering.Harness

@testset "Eval.Metrics" begin
    @testset "parsing_accuracy — exact-match semantics" begin
        @test parsing_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0
        @test parsing_accuracy(["a", "b", "d"], ["a", "b", "c"]) ≈ 2/3
        @test parsing_accuracy(String[], String[]) == 0.0
        @test_throws DimensionMismatch parsing_accuracy(["a"], ["a", "b"])
    end

    @testset "group_accuracy — multiset membership" begin
        # pred and gold partition the 6 items the same way.
        pred = ["x", "x", "y", "y", "z", "z"]
        gold = ["a", "a", "b", "b", "c", "c"]
        @test group_accuracy(pred, gold) == 1.0
        # Over-splitting: gold has one cluster, pred splits it in two.
        @test group_accuracy(["x", "x", "y", "y"], ["a", "a", "a", "a"]) == 0.0
        # Partial overlap (line 4 goes to the wrong pred group).
        @test group_accuracy(["x", "x", "y", "x"], ["a", "a", "a", "a"]) ≈ 0.0
    end

    @testset "grouping_f1 — perfect, split, merge" begin
        # Perfect matching.
        pred = ["x", "x", "y", "y"]
        gold = ["a", "a", "b", "b"]
        p, r, f = grouping_f1(pred, gold)
        @test (p, r, f) == (1.0, 1.0, 1.0)
        # Over-split: pred singletons, gold one cluster → 0 pairs in common.
        p, r, f = grouping_f1(["x", "y", "z"], ["a", "a", "a"])
        @test p == 0.0 && r == 0.0 && f == 0.0
        # Over-merge: pred one cluster, gold singletons → 0 TP too.
        p, r, f = grouping_f1(["x", "x", "x"], ["a", "b", "c"])
        @test p == 0.0 && r == 0.0 && f == 0.0
    end

    @testset "template_group_f1 — set-match semantics" begin
        pred = ["x", "x", "y", "y"]
        gold = ["a", "a", "b", "b"]
        p, r, f = template_group_f1(pred, gold)
        @test (p, r, f) == (1.0, 1.0, 1.0)
        # One pred template is correct, one isn't.
        pred = ["x", "x", "y", "z"]
        gold = ["a", "a", "b", "b"]
        p, r, f = template_group_f1(pred, gold)
        @test p ≈ 1/3       # {1,2} matches {1,2}, the singletons don't
        @test r ≈ 0.5
    end

    @testset "nmi / ari / purity / v_measure — perfect and worst cases" begin
        pred = ["a", "a", "b", "b"]
        gold = ["x", "x", "y", "y"]
        @test nmi(pred, gold) ≈ 1.0
        @test ari(pred, gold) == 1.0
        @test purity(pred, gold) == 1.0
        h, c, v = v_measure(pred, gold)
        @test h == 1.0 && c == 1.0 && v == 1.0

        # Uninformative prediction: every item in one cluster.
        pred = fill("x", 4)
        gold = ["a", "a", "b", "b"]
        @test nmi(pred, gold) == 0.0
        # ARI for a degenerate clustering is 0 (chance-level).
        @test ari(pred, gold) == 0.0
        # Purity still reflects majority-class concentration (50 %).
        @test purity(pred, gold) == 0.5
    end

    @testset "fmi (Fowlkes-Mallows)" begin
        # Perfect partition.
        @test fmi(["a", "a", "b", "b"], ["x", "x", "y", "y"]) == 1.0

        # Worked example: pred={{1,2},{3,4,5}} vs gold={{1,2,3},{4,5}}.
        # Same-pred pairs   = C(2,2) + C(3,2) = 1 + 3 = 4
        # Same-gold pairs   = C(3,2) + C(2,2) = 3 + 1 = 4
        # TP (same both)    = C(2,2) [{1,2}] + C(2,2) [{4,5}] = 1 + 1 = 2
        # FMI = 2 / sqrt(4 * 4) = 0.5
        @test fmi([1, 1, 2, 2, 2], [1, 1, 1, 2, 2]) ≈ 0.5 atol = 1e-12

        # Degenerate "one big cluster" — no pairs discriminated.
        # Same-pred pairs = C(4,2) = 6. Same-gold pairs = 2·C(2,2) = 2.
        # TP = 2 (pairs {1,2} and {3,4} share both clusters trivially).
        # FMI = 2 / sqrt(6·2) = 2/sqrt(12) ≈ 0.577.
        @test fmi(fill("x", 4), ["a", "a", "b", "b"]) ≈
              2 / sqrt(12) atol = 1e-10

        # FMI ranges in [0, 1]; a clearly-worse partition scores lower.
        good = fmi([1, 1, 2, 2], [1, 1, 2, 2])           # perfect = 1
        bad  = fmi([1, 2, 1, 2], [1, 1, 2, 2])           # anti-aligned
        @test good > bad
        @test 0.0 <= bad <= 1.0

        # Symmetry.
        @test fmi([1, 1, 2, 2], [1, 1, 2, 2]) ==
              fmi([1, 1, 2, 2], [1, 1, 2, 2])

        # Dimension mismatch.
        @test_throws DimensionMismatch fmi([1, 2], [1, 2, 3])
    end

    @testset "to_labels — stable first-seen encoding" begin
        @test to_labels(["b", "a", "a", "c", "b"]) == [1, 2, 2, 3, 1]
    end
end

@testset "Eval.Harness" begin
    path = joinpath(@__DIR__, "..", "benchmarks", "loghub2", "toy_structured.csv")
    dataset = load_loghub(path)
    @test length(dataset) == 6
    @test dataset.templates[1] == "INFO started pid <*>"
    @test dataset.event_ids[4] == "E2"

    # Identity parser — one template per line.
    r = run_parser(dataset, ls -> [String(l) for l in ls]; parser_name = "identity")
    @test r.n == 6
    @test r.parser == "identity"
    @test r.fta[3] < 1.0                           # oversplits into 6 templates

    # Ground-truth parser — cheats by returning the gold templates.
    # The harness must give it perfect scores across every metric.
    r_gt = run_parser(dataset, _ -> copy(dataset.templates); parser_name = "gold")
    @test r_gt.pa == 1.0
    @test r_gt.ga == 1.0
    @test r_gt.nmi ≈ 1.0
    @test r_gt.ari == 1.0
    @test r_gt.purity == 1.0
    @test r_gt.v == 1.0

    # Parser that returns the wrong number of templates must error.
    @test_throws DimensionMismatch run_parser(dataset, _ -> ["x", "y"];
                                              parser_name = "broken")
end
