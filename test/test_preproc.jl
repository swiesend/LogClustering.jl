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

    @testset "ranking puts complex patterns ahead of INT/DECIMAL" begin
        # IP must beat dotted integer groups.
        @test mask_line("from 10.0.0.1 sent") == "from <IP> sent"
        # Timestamp wins over any numeric.
        @test occursin("<TIMESTAMP>",
                      mask_line("ts=2024-01-01T00:00:00Z rest"))
    end

    @testset "INT vs DECIMAL split (EN dot, DE comma, optional thousands)" begin
        # Plain integers land in INT regardless of locale grouping.
        @test mask_line("count 42") == "count <INT>"
        @test mask_line("count 1,234") == "count <INT>"
        @test mask_line("count 1.234") == "count <INT>"      # DE grouping of plain int
        # Decimals go to the locale-specific slot.
        @test mask_line("pi 3.14") == "pi <DECIMAL_EN>"
        @test mask_line("pi 3,14") == "pi <DECIMAL_DE>"
        @test mask_line("total 1,234.56") == "total <DECIMAL_EN>"
        @test mask_line("total 1.234,56") == "total <DECIMAL_DE>"
        # Signed decimals.
        @test mask_line("delta -0.5") == "delta <DECIMAL_EN>"
        @test mask_line("delta -0,5") == "delta <DECIMAL_DE>"
    end

    @testset "template customisation" begin
        @test mask_line("user 42 ok"; template = "%\$LABEL%") ==
              "user %INT% ok"
    end

    @testset "argument validation" begin
        @test_throws ArgumentError mask_line("x"; labels = ["A"], patterns = String[])
    end

    @testset "mask_lines — vector form" begin
        lines = ["user 10.0.0.1 ok", "user 10.0.0.2 ok"]
        out = mask_lines(lines)
        @test out == ["user <IP> ok", "user <IP> ok"]
    end

    @testset "mask_line_with_values — captures slot values alongside template" begin
        tpl, vals = Masking.mask_line_with_values("user 42 from 10.0.0.1")
        @test tpl == "user <INT> from <IP>"
        @test length(vals) == 2
        @test vals[1].label == "INT" && vals[1].value == "42"
        @test vals[2].label == "IP"  && vals[2].value == "10.0.0.1"
        @test vals[1].start == findfirst("42", "user 42 from 10.0.0.1")[1]
    end

    @testset "mask_lines_with_values — parallel template + value vectors" begin
        lines = ["user 10.0.0.1 ok", "user 10.0.0.2 ok"]
        tpls, vals = Masking.mask_lines_with_values(lines)
        @test tpls == ["user <IP> ok", "user <IP> ok"]
        @test length(vals) == 2
        @test vals[1][1].value == "10.0.0.1"
        @test vals[2][1].value == "10.0.0.2"
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

    @testset "LRUMemo — memoize + FIFO eviction + counters" begin
        m = Dedup.LRUMemo{String, Float64}(3)
        calls = Ref(0)
        f(k) = Dedup.memoize!(m, k, () -> (calls[] += 1; Float64(length(k))))
        @test f("aa")  == 2.0            # miss → compute
        @test f("aa")  == 2.0            # hit  → cached
        @test f("bbb") == 3.0            # miss
        @test calls[]  == 2              # only two computes
        @test m.hits == 1 && m.misses == 2
        # Fill past capacity 3 → oldest ("aa") evicted.
        f("c"); f("dddd")
        @test length(m) == 3
        @test !haskey(m, "aa")
        @test haskey(m, "dddd")
    end
end
