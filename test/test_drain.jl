using Test
using LogClustering
using LogClustering.Drain3
using LogClustering.Drain3: Drain, template_of, process!, parse_all

@testset "Parsers.Drain" begin
    @testset "construction — validation" begin
        @test_throws ArgumentError Drain(depth = 1)
        @test_throws ArgumentError Drain(sim_th = 1.5)
        @test_throws ArgumentError Drain(sim_th = -0.1)
    end

    @testset "one template absorbs a repeated structure" begin
        d = Drain(; depth = 4, sim_th = 0.4)
        lines = [
            "INFO started pid 1001",
            "INFO started pid 1002",
            "INFO started pid 1003",
        ]
        ids = Int[]
        tpls = String[]
        for l in lines
            cid, t = process!(d, l)
            push!(ids, cid); push!(tpls, t)
        end
        # All three land in the same cluster, and the variable pid is
        # generalised after the second line.
        @test all(==(ids[1]), ids)
        @test tpls[3] == "INFO started pid <*>"
        @test d.clusters[1].size == 3
    end

    @testset "different event types form separate clusters" begin
        d = Drain()
        lines = [
            "INFO started pid 1001",
            "INFO started pid 1002",
            "ERROR connection refused from 10.0.0.1",
            "ERROR connection refused from 10.0.0.2",
            "INFO service ready",
        ]
        ids = [first(process!(d, l)) for l in lines]
        @test ids[1] == ids[2]
        @test ids[3] == ids[4]
        @test ids[1] != ids[3]
        @test ids[5] ∉ (ids[1], ids[3])
        @test length(d.clusters) == 3
    end

    @testset "parse_all returns one template per line" begin
        d = Drain()
        lines = [
            "INFO started pid 1001",
            "INFO started pid 1002",
            "ERROR connection refused from 10.0.0.1",
        ]
        tpls = parse_all(d, lines)
        @test length(tpls) == 3
        @test tpls[1] == tpls[2]
        @test tpls[3] != tpls[1]
    end

    @testset "parametrize hook — numeric-token heuristic routes numbers via <*>" begin
        # The numeric token sits *at a routing position* (token 2 at
        # depth = 4). With the default parametrize predicate, `1001`
        # and `1002` both route via `<*>` and share a leaf — one
        # cluster. Disable parametrize and each integer lands in its
        # own literal branch, so we get two clusters.
        lines = ["INFO 1001 ok", "INFO 1002 ok"]
        d_num = Drain(; depth = 4)
        parse_all(d_num, lines)
        @test length(d_num.clusters) == 1

        d_lit = Drain(; depth = 4, parametrize = _ -> false)
        parse_all(d_lit, lines)
        @test length(d_lit.clusters) == 2
    end

    @testset "sim_th — low threshold merges eagerly; high threshold splits" begin
        lines = [
            "INFO ok alice",
            "INFO ok bob",
            "INFO ok carol",
        ]
        d_lo = Drain(; sim_th = 0.2)
        parse_all(d_lo, lines)
        @test length(d_lo.clusters) == 1               # all merge

        d_hi = Drain(; sim_th = 0.99)
        parse_all(d_hi, lines)
        @test length(d_hi.clusters) >= 2               # forced to split
    end

    @testset "stable behaviour across re-run when state is fresh" begin
        lines = [
            "INFO started pid 1001",
            "ERROR 500 from 10.0.0.5",
            "INFO started pid 1002",
            "DEBUG heartbeat 17",
        ]
        tpls1 = parse_all(Drain(), lines)
        tpls2 = parse_all(Drain(), lines)
        @test tpls1 == tpls2
    end

    @testset "empty / whitespace-only lines are handled without crashing" begin
        d = Drain()
        cid, t = process!(d, "")
        @test cid == 0 && t == ""
        cid, t = process!(d, "   ")
        @test cid == 0 && t == ""
    end

    @testset "max_clusters — LRU eviction bounds the cluster set" begin
        d = Drain(; max_clusters = 3, sim_th = 0.95)
        # Five clearly-distinct templates; only 3 may survive.
        for s in ("alpha aa bb", "beta cc dd", "gamma ee ff",
                  "delta gg hh", "epsilon ii jj")
            process!(d, s)
        end
        @test length(d.clusters) == 3            # bounded, not 5
        # last_seen / owner stay parallel to clusters.
        @test length(d.last_seen) == 3
        @test length(d.owner) == 3
    end

    @testset "max_clusters — recently-seen clusters survive eviction" begin
        d = Drain(; max_clusters = 2, sim_th = 0.95)
        process!(d, "keep me alpha")             # slot A
        process!(d, "other beta gamma")          # slot B (cap reached)
        process!(d, "keep me alpha")             # bump A's recency
        process!(d, "brand new delta")           # new → evict LRU (B, not A)
        templates = [template_of(c) for c in d.clusters]
        @test length(d.clusters) == 2
        @test any(t -> occursin("keep", t), templates)   # A survived
        @test any(t -> occursin("delta", t), templates)  # new one present
        @test !any(t -> occursin("beta", t), templates)  # B evicted
    end

    @testset "max_clusters — cap with an empty current leaf doesn't throw" begin
        # The old `first(node.clusters)` stub threw when the routing leaf
        # for a new line had no clusters yet. Eviction must handle it.
        d = Drain(; max_clusters = 1, sim_th = 0.99, depth = 2)
        process!(d, "aaa bbb ccc")
        # A line routing to a different leaf while at the cap.
        @test_nowarn process!(d, "zzz yyy xxx www vvv")
        @test length(d.clusters) == 1
    end

    @testset "max_clusters = 0 (default) leaves growth unbounded" begin
        d = Drain(; sim_th = 0.99)
        for i in 1:20
            process!(d, "template number $i alpha beta")
        end
        @test length(d.clusters) >= 15           # no eviction
    end

    @testset "rebuild_lru! restores parallel arrays after a load" begin
        d = Drain(; max_clusters = 5)
        for s in ("one aa", "two bb", "three cc")
            process!(d, s)
        end
        # Simulate a fresh-load state: wipe the non-persisted bookkeeping.
        d.last_seen = Int[]
        d.owner = Drain3.TreeNode[]
        d.access_tick = 0
        Drain3.rebuild_lru!(d)
        @test length(d.last_seen) == length(d.clusters)
        @test length(d.owner) == length(d.clusters)
        # And the parser keeps working (eviction path reachable).
        @test process!(d, "four dd")[1] > 0
    end
end
