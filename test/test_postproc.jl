using Test
using LogClustering
using LogClustering.Purity
using LogClustering.Typing
using LogClustering.Merge

@testset "PostProc.Purity" begin
    @testset "slot_entropy" begin
        @test slot_entropy(String[]) == 0.0
        @test slot_entropy(["a"]) == 0.0
        @test slot_entropy(["a", "a", "a"]) == 0.0
        # Uniform over 2 symbols → 1 bit.
        @test slot_entropy(["a", "b"]) ≈ 1.0
        # Uniform over 4 symbols → 2 bits.
        @test slot_entropy(["a", "b", "c", "d"]) ≈ 2.0
        # Non-uniform: 3/4 vs 1/4.
        h = slot_entropy(["a", "a", "a", "b"])
        @test 0.8 < h < 0.82      # ≈ 0.8113
    end

    @testset "slot_realisations" begin
        tpl = "user <*> from <*>"
        lines = [
            "user 1 from host_a",
            "user 2 from host_b",
            "user 3 from host_a",
        ]
        reals = slot_realisations(tpl, lines)
        @test length(reals) == 2
        @test reals[1] == ["1", "2", "3"]
        @test reals[2] == ["host_a", "host_b", "host_a"]
    end

    @testset "slot_realisations — length-mismatched lines skipped" begin
        tpl = "a <*> c"
        lines = ["a 1 c", "totally different", "a 2 c extra"]
        reals = slot_realisations(tpl, lines)
        @test reals[1] == ["1"]
    end

    @testset "promote_pure_slots — constant slot becomes literal" begin
        tpl = "INFO <*> request <*>"
        lines = [
            "INFO auth request 42",
            "INFO auth request 43",
            "INFO auth request 44",
        ]
        out = promote_pure_slots(tpl, lines)
        # First slot is constant "auth"; second is variable.
        @test out == "INFO auth request <*>"
    end

    @testset "promote_pure_slots — both variable stay as <*>" begin
        tpl = "<*> <*>"
        lines = ["a 1", "b 2", "c 3"]
        out = promote_pure_slots(tpl, lines)
        @test out == "<*> <*>"
    end

    @testset "promote_pure_slots — τ tunes how strict 'pure' is" begin
        tpl = "<*>"
        # 3/4 identical → entropy ≈ 0.81 bits. τ=0.2 keeps it as <*>;
        # τ=0.9 promotes it to the majority value.
        lines = ["foo", "foo", "foo", "bar"]
        @test promote_pure_slots(tpl, lines; τ = 0.2) == "<*>"
        @test promote_pure_slots(tpl, lines; τ = 0.9) == "foo"
    end

    @testset "promote_pure_slots — no placeholder is a no-op" begin
        @test promote_pure_slots("plain literal", ["plain literal"]) == "plain literal"
    end

    @testset "promote_pure_slots — typed placeholder matched too" begin
        tpl = "host <IP> port <INT>"
        lines = ["host 10.0.0.1 port 80", "host 10.0.0.2 port 80"]
        # IP varies, port is constant.
        @test promote_pure_slots(tpl, lines) == "host <IP> port 80"
    end
end

@testset "PostProc.Typing" begin
    @testset "type_slot — IP battery" begin
        @test type_slot(["10.0.0.1", "192.168.1.1", "8.8.8.8"]) == "<IP>"
    end

    @testset "type_slot — INT" begin
        @test type_slot(["1", "42", "9999"]) == "<INT>"
    end

    @testset "type_slot — mixed types fall back to <*>" begin
        @test type_slot(["10.0.0.1", "hello", "42"]) == "<*>"
    end

    @testset "type_slot — empty input → wildcard" begin
        @test type_slot(String[]) == "<*>"
    end

    @testset "type_slot — min_match_fraction gate" begin
        vals = ["10.0.0.1", "10.0.0.2", "not an ip"]  # 2/3 ≈ 0.67
        @test type_slot(vals; min_match_fraction = 0.95) == "<*>"
        @test type_slot(vals; min_match_fraction = 0.5)  == "<IP>"
    end

    @testset "type_template — promote <*> slots to types" begin
        tpl = "host <*> port <*>"
        lines = [
            "host 10.0.0.1 port 80",
            "host 10.0.0.2 port 443",
            "host 10.0.0.3 port 22",
        ]
        out = type_template(tpl, lines)
        @test out == "host <IP> port <INT>"
    end

    @testset "type_template — no-placeholder template is unchanged" begin
        @test type_template("plain literal", ["plain literal"]) == "plain literal"
    end

    @testset "type_template — slots without a clear type stay <*>" begin
        tpl = "mixed <*>"
        lines = ["mixed alpha", "mixed beta", "mixed gamma"]
        @test type_template(tpl, lines) == "mixed <*>"
    end
end

@testset "PostProc.Merge" begin
    @testset "token_edit_distance" begin
        @test token_edit_distance("a b c", "a b c") == 0
        @test token_edit_distance("a b c", "a b d") == 1   # substitute
        @test token_edit_distance("a b c", "a b c d") == 1 # insert
        @test token_edit_distance("a b c", "a c") == 1     # delete
        @test token_edit_distance("", "a b") == 2
        @test token_edit_distance("a b", "") == 2
        @test token_edit_distance("", "") == 0
    end

    @testset "merge_pairs — finds within-distance candidates" begin
        templates = [
            "INFO user <*> login",
            "INFO user <*> logout",   # 1 substitution
            "ERROR disk <*> failed",  # distant
        ]
        pairs = merge_pairs(templates; max_token_distance = 1)
        @test (1, 2) in pairs
        @test !((1, 3) in pairs)
        @test !((2, 3) in pairs)
    end

    @testset "merge_pairs — wildcard-diff guard" begin
        # Two templates within edit distance 1: one has 2 wildcards,
        # the other has 1 (a literal substituted in for the slot).
        templates = ["a <*> b <*> c", "a x b <*> c"]
        @test isempty(merge_pairs(templates; max_token_distance = 1,
                                  max_wildcard_diff = 0))
        # Relax the wildcard guard and the pair appears.
        @test (1, 2) in merge_pairs(templates; max_token_distance = 1,
                                    max_wildcard_diff = 1)
    end

    @testset "merge_clusters — remaps IDs for merged groups" begin
        templates = [
            "INFO user <*> login",
            "INFO user <*> logout",
            "ERROR disk failed",
            "INFO user <*> login",   # duplicate of cluster 1
        ]
        assignments = [1, 2, 3, 4]
        new_assign, new_templates = merge_clusters(assignments, templates;
                                                   max_token_distance = 1)
        # Lines 1, 2, 4 share the surviving template (lex-smallest of
        # "INFO user <*> login" / "INFO user <*> logout").
        @test new_assign[1] == new_assign[2] == new_assign[4]
        @test new_assign[3] != new_assign[1]
        @test length(unique(new_assign)) == 2
        @test length(new_templates) == 2
        # Lex-smallest of "login" and "logout" is "login".
        @test "INFO user <*> login" in new_templates
        @test "ERROR disk failed" in new_templates
    end

    @testset "merge_clusters — no merges leaves assignments intact" begin
        templates = ["A <*>", "B <*>", "C <*>"]   # all pairwise distance > 1 after canonicalisation
        assignments = [1, 2, 3]
        new_assign, new_templates = merge_clusters(assignments, templates;
                                                   max_token_distance = 0)
        @test length(unique(new_assign)) == 3
        @test length(new_templates) == 3
    end

    @testset "merge_clusters — dimension check" begin
        @test_throws DimensionMismatch merge_clusters([1, 2], ["a", "b", "c"])
    end

    @testset "merge_clusters — transitive closure via union-find" begin
        # A–B within 1, B–C within 1, A–C within 2 → all three collapse.
        templates = ["a b c", "a b d", "a e d"]
        assignments = [1, 2, 3]
        new_assign, new_templates = merge_clusters(assignments, templates;
                                                   max_token_distance = 1,
                                                   max_wildcard_diff = 2)
        @test new_assign[1] == new_assign[2] == new_assign[3]
        @test length(new_templates) == 1
    end
end
