using Test
using Random
using LogClustering
using LogClustering.Masking
using LogClustering.Masking: DEFAULT_LABELS, DEFAULT_PATTERNS, mask_line, mask_lines
using LogClustering.Dedup
using LogClustering.Dedup: DedupState, is_new!, dedup, fpr_estimate

@testset "PreProc.Masking" begin
    @testset "default battery masks the common shapes" begin
        @test mask_line("2024-04-17T10:00:00Z host 10.0.0.1 ok") ==
              "<TIMESTAMP> host <IP> ok"
        @test mask_line("session 550e8400-e29b-41d4-a716-446655440000 ok") ==
              "session <UUID> ok"
        @test mask_line("request 23ms") == "request <DURATION>"
        @test mask_line("bytes 4.5KB") == "bytes <SIZE>"
    end

    @testset "ranking puts complex patterns ahead of NUM" begin
        # IP must beat NUM — four dotted octets must *not* be masked as
        # four separate <NUM>s.
        @test mask_line("from 10.0.0.1 sent") == "from <IP> sent"
        # Timestamp must beat NUM too.
        @test occursin("<TIMESTAMP>",
                      mask_line("ts=2024-01-01T00:00:00Z rest"))
    end

    @testset "template customisation" begin
        @test mask_line("user 42 ok"; template = "%\$LABEL%") ==
              "user %NUM% ok"
    end

    @testset "argument validation" begin
        @test_throws ArgumentError mask_line("x"; labels = ["A"], patterns = String[])
    end

    @testset "mask_lines — vector form" begin
        lines = ["user 10.0.0.1 ok", "user 10.0.0.2 ok"]
        out = mask_lines(lines)
        @test out == ["user <IP> ok", "user <IP> ok"]
    end
end

@testset "PreProc.Dedup" begin
    @testset "DedupState — sizing via Bloom formula" begin
        s = DedupState(; expected_n = 1000, fpr = 1e-4)
        @test s.m >= 8
        @test s.k >= 1
        @test length(s.bits) == s.m
    end

    @testset "DedupState — rejects silly inputs" begin
        @test_throws ArgumentError DedupState(; expected_n = 0)
        @test_throws ArgumentError DedupState(; expected_n = 10, fpr = 0)
        @test_throws ArgumentError DedupState(; expected_n = 10, fpr = 1)
    end

    @testset "is_new! — first-seen vs repeated" begin
        s = DedupState(; expected_n = 100)
        @test is_new!(s, "a")
        @test is_new!(s, "b")
        @test !is_new!(s, "a")
        @test !is_new!(s, "b")
        @test is_new!(s, "c")
        @test s.observed == 5
    end

    @testset "dedup — preserves first-seen order" begin
        lines = ["a", "b", "a", "c", "b", "d"]
        uniq, mask = dedup(lines)
        @test uniq == ["a", "b", "c", "d"]
        @test mask == BitVector([1, 1, 0, 1, 0, 1])
    end

    @testset "fpr estimate grows with fill" begin
        s = DedupState(; expected_n = 100, fpr = 1e-4)
        @test fpr_estimate(s) == 0.0
        for i in 1:50
            is_new!(s, i)
        end
        f = fpr_estimate(s)
        @test 0 < f < 1
    end

    @testset "Bloom false-positive rate respects the configured bound" begin
        # Insert N items, then query-only on 10 000 disjoint probes.
        # Must use the pure `contains` query here — `is_new!` would
        # grow the filter on every probe and distort the measurement.
        function measure_fpr(rng_seed, N, probes, configured)
            rng = Random.MersenneTwister(rng_seed)
            s = DedupState(; expected_n = N, fpr = configured)
            observed = Set{String}()
            for _ in 1:N
                w = string("obs-", rand(rng, 1:10^9))
                push!(observed, w)
                is_new!(s, w)
            end
            fps = 0
            for i in 1:probes
                w = string("probe-", i)
                w in observed && continue
                if Dedup.contains(s, w)
                    fps += 1
                end
            end
            return fps / probes
        end
        # Allow 3× the configured rate for the finite-filter tail.
        @test measure_fpr(11, 1000, 10_000, 1e-2) < 0.03
    end
end
