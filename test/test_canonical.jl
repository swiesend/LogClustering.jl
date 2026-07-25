using Test
using LogClustering
using LogClustering.Canonical
using LogClustering.Canonical: WILDCARD

@testset "PostProc.Canonical" begin
    @testset "typed markers + regex fragments collapse to <*>" begin
        @test canonicalise("INFO <IP> ok")   == "INFO <*> ok"
        @test canonicalise("INFO <INT> ok")  == "INFO <*> ok"
        @test canonicalise("INFO .*? done")  == "INFO <*> done"
        @test canonicalise("INFO .+? done")  == "INFO <*> done"
        @test canonicalise("thesis %LABEL% x") == "thesis <*> x"
    end

    @testset "consecutive wildcards collapse" begin
        @test canonicalise("a <*> <*> b") == "a <*> b"
        @test canonicalise("a <*> <*> <*> b") == "a <*> b"
        @test canonicalise("a <IP> <NUM> b") == "a <*> b"
    end

    @testset "whitespace normalised" begin
        @test canonicalise("a   b\t\tc") == "a b c"
        @test canonicalise("  padded  ") == "padded"
    end

    @testset "alternations sort + dedup" begin
        @test canonicalise("(b|a|c)") == "(a|b|c)"
        @test canonicalise("status (c|a|b|a)") == "status (a|b|c)"
    end

    @testset "opt-out keywords" begin
        # Preserve typed markers when asked.
        @test canonicalise("INFO <IP> ok"; collapse_typed_markers = false) ==
              "INFO <IP> ok"
        # Preserve original whitespace.
        @test canonicalise("a   b"; normalise_whitespace = false) == "a   b"
    end

    @testset "canonical_hash — equal canonical forms share hash" begin
        @test canonical_hash("INFO <IP> ok") == canonical_hash("INFO <*> ok")
        @test canonical_hash("a b c")         != canonical_hash("a b d")
    end

    @testset "canonicalise_all — duplicate templates merge" begin
        templates = ["INFO <IP> ok", "INFO <*> ok", "ERROR <*>"]
        out = canonicalise_all(templates)
        @test out == ["INFO <*> ok", "INFO <*> ok", "ERROR <*>"]
        # The first two land in the same group when we use these as
        # cluster labels downstream:
        @test length(unique(out)) == 2
    end
end
