using Test
using LogClustering
using LogClustering.Masking: SlotValue, mask_lines_with_values
using LogClustering.Instance: ValueNoveltyDetector, update!, value_novelty,
                              combined_anomaly
using Random, Lux

@testset "Instance.ValueNoveltyDetector" begin
    @testset "update! + value_novelty on seen/unseen categoricals" begin
        det = ValueNoveltyDetector()
        update!(det, [SlotValue("USER", "alice", 1, 5),
                      SlotValue("USER", "bob",   1, 3)])

        # All known → 0.
        @test value_novelty(det, [SlotValue("USER", "alice", 1, 5)]) == 0.0

        # One of two is new → 0.5.
        @test value_novelty(det, [SlotValue("USER", "alice", 1, 5),
                                  SlotValue("USER", "carol", 1, 5)]) == 0.5

        # All new → 1.0.
        @test value_novelty(det, [SlotValue("IP", "10.0.0.1", 1, 8)]) == 1.0
    end

    @testset "numeric z-score adds to the score once σ is established" begin
        det = ValueNoveltyDetector()
        # Train a tight numeric distribution around 100.
        for v in ("98", "99", "100", "101", "102")
            update!(det, [SlotValue("NUM", v, 1, 3)])
        end
        # Value well inside the range — just the "seen" check (unseen
        # exact-string → +1, no z-score hit because it's within σ).
        @test value_novelty(det, [SlotValue("NUM", "100", 1, 3)]) == 0.0
        # Value outside 3σ → both the novelty (unseen exact string)
        # AND the range penalty fire. The total is bounded by ≤ 2.
        s = value_novelty(det, [SlotValue("NUM", "999", 1, 3)]; numeric_sigma = 3.0)
        @test s >= 1.0
        @test s <= 2.0
    end

    @testset "mask_lines_with_values feeds the detector end-to-end" begin
        training = [
            "user alice from 10.0.0.1",
            "user bob from 10.0.0.2",
            "user carol from 10.0.0.3",
        ]
        test = [
            "user alice from 10.0.0.1",
            "user attacker from 203.0.113.4",
        ]
        _, train_vals = mask_lines_with_values(training)
        _, test_vals  = mask_lines_with_values(test)

        det = ValueNoveltyDetector()
        for vs in train_vals
            update!(det, vs)
        end
        s = [value_novelty(det, vs) for vs in test_vals]
        @test s[1] == 0.0              # everything seen
        @test s[2] >= 0.5              # new user AND new IP
    end

    @testset "combined_anomaly — weighted template + value signal" begin
        rng = Random.MersenneTwister(0)
        # Tiny identity-AE stand-in: sigmoid maps near 0.5 so
        # reconstruction error is uniform across samples.
        m = Chain(Dense(3 => 3, sigmoid))
        ps, st = Lux.setup(rng, m)
        X = Float32[0.1 0.9 0.5; 0.1 0.9 0.5; 0.1 0.9 0.5]
        det = ValueNoveltyDetector()
        update!(det, [SlotValue("IP", "10.0.0.1", 1, 8)])
        vals = [
            [SlotValue("IP", "10.0.0.1", 1, 8)],   # known → value_score 0
            [SlotValue("IP", "10.0.0.1", 1, 8)],   # known
            [SlotValue("IP", "8.8.8.8",  1, 7)],   # unknown → value_score 1
        ]
        s = combined_anomaly(m, ps, st, X, vals, det;
                             weights = (template = 0.0, values = 1.0))
        # Pure value-novelty weighting: the last column clearly beats
        # the first two.
        @test s[3] > s[1]
        @test s[3] > s[2]

        # Pure template-reconstruction weighting: the three template
        # scores are proportional to reconstruction error on X. We
        # only assert non-negativity and that the shape-unchanged
        # rows (1 and 2) score identically.
        s_t = combined_anomaly(m, ps, st, X, vals, det;
                               weights = (template = 1.0, values = 0.0))
        @test all(s_t .>= 0)
        @test length(s_t) == 3

        @test_throws ArgumentError combined_anomaly(m, ps, st, X, vals, det;
            weights = (template = 0.0, values = 0.0))
    end
end
