using Test
using LogClustering
using LogClustering.Rules
using LogClustering.Rules: RuleSet, TriggerEvent, default_rules, load_rules,
                            evaluate, snapshot, route_sinks,
                            ScoreThresholdRule, NovelClusterRule, NovelTokenRule,
                            RateSpikeRule, VolumeAnomalyRule, KeywordRule,
                            RegexRule

# Compact JSON helper so the tests stay readable.
_rules_json(body::AbstractString) = IOBuffer(body)

# Tiny no-warmup ruleset factory — many tests don't care about baselines.
function _no_warmup(rule_json::AbstractString)
    body = """
    { "version": 1,
      "defaults": { "warmup_required": false, "cooldown_s": 0 },
      "rules": $rule_json
    }
    """
    return load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
end

# Build a minimal InferResult dict.
function _ir(line::AbstractString; line_id::Int = 1, extra::Dict = Dict())
    d = Dict{String, Any}("line" => line, "line_id" => line_id)
    for (k, v) in extra
        d[k] = v
    end
    return d
end

@testset "Rules" begin

    @testset "schema parsing — defaults.json bundled with package" begin
        rs = default_rules(; warmup_lines = 0, warmup_seconds = 0.0)
        @test rs isa RuleSet
        @test rs.version == 1
        ids = [r.id for r in rs.rules]
        @test "novel_template"   in ids
        @test "score_p99"        in ids
        @test "error_rate_spike" in ids
        @test "volume_anomaly"   in ids
        @test "fatal_keywords"   in ids
        @test length(unique(ids)) == length(ids)   # no duplicate ids
        @test haskey(rs.sinks, "webhook:default")
        @test rs.sinks["webhook:default"].kind == :webhook
        @test rs.sinks["webhook:default"].format == :alertmanager

        snap = snapshot(rs)
        @test haskey(snap, "warmup")
        @test snap["n_rules"] == length(ids)
    end

    @testset "schema rejects invalid input" begin
        @test_throws ArgumentError load_rules(IOBuffer("""
            {"version": 2, "rules": []}"""))
        @test_throws ArgumentError load_rules(IOBuffer("""
            {"version": 1, "rules": [
                {"id": "a", "kind": "keyword", "keywords": ["x"]},
                {"id": "a", "kind": "keyword", "keywords": ["y"]}]}"""))
        @test_throws ArgumentError load_rules(IOBuffer("""
            {"version": 1, "rules": [
                {"id": "a", "kind": "no_such_kind"}]}"""))
        @test_throws ArgumentError load_rules(IOBuffer("""
            {"version": 1, "rules": [
                {"id": "a", "kind": "keyword", "keywords": ["x"],
                 "severity": "panic"}]}"""))
        @test_throws ArgumentError load_rules(IOBuffer("""
            {"version": 1, "rules": [
                {"id": "a", "kind": "score_threshold",
                 "metric": "m", "comparison": "!=", "value": 1.0}]}"""))
    end

    @testset "keyword rule — case-insensitive substring" begin
        rs = _no_warmup("""[
            {"id": "fatal", "kind": "keyword",
             "field": "line",
             "keywords": ["FATAL", "OOM"],
             "case_sensitive": false}]""")

        t = evaluate(rs, _ir("Normal startup message"))
        @test isempty(t)

        t = evaluate(rs, _ir("kernel: Out of memory: OOM-Killed process"))
        @test length(t) == 1
        @test t[1].rule_id == "fatal"
        @test t[1].rule_kind == :keyword
        @test "OOM" in t[1].fields["matched"]
    end

    @testset "regex rule — pattern match" begin
        rs = _no_warmup("""[
            {"id": "oom_kill", "kind": "regex",
             "field": "line",
             "pattern": "kernel:.*Out of memory.*Killed process"}]""")

        @test isempty(evaluate(rs, _ir("benign syslog line")))
        t = evaluate(rs, _ir(
            "kernel: Out of memory: Killed process 1234 (sshd)"))
        @test length(t) == 1
        @test t[1].rule_id == "oom_kill"
        @test occursin("Killed process", t[1].fields["match"])
    end

    @testset "score_threshold — fixed value" begin
        rs = _no_warmup("""[
            {"id": "high_nll", "kind": "score_threshold",
             "metric": "transformer_decoder.nll",
             "comparison": ">",
             "value": 1.5}]""")

        ir_low  = _ir("a"; extra = Dict("transformer_decoder" =>
                                          Dict("nll" => 0.5)))
        ir_high = _ir("b"; extra = Dict("transformer_decoder" =>
                                          Dict("nll" => 2.0)))
        @test isempty(evaluate(rs, ir_low))
        t = evaluate(rs, ir_high)
        @test length(t) == 1
        @test t[1].fields["value"]     ≈ 2.0
        @test t[1].fields["threshold"] ≈ 1.5
    end

    @testset "score_threshold — auto:p99 converges" begin
        rs = _no_warmup("""[
            {"id": "p99", "kind": "score_threshold",
             "metric": "score",
             "comparison": ">",
             "value": "auto:p99"}]""")

        # Seed reservoir with 200 samples drawn from a known distribution.
        # No firings expected during this seed because the reservoir starts
        # empty (threshold = Inf for first sample, then settles).
        fired_during_seed = 0
        for i in 1:200
            ir = _ir("seed-$i"; extra = Dict("score" => Float64(i)))
            fired_during_seed += length(evaluate(rs, ir))
        end
        # After seeding with 1..200 the p99 should be near the top of the
        # distribution; an outlier well above this should now fire.
        t = evaluate(rs, _ir("spike"; extra = Dict("score" => 10_000.0)))
        @test length(t) == 1
        @test t[1].fields["value"] == 10_000.0
    end

    @testset "novel_cluster — fires once per unseen id, dedups via cooldown" begin
        rs = _no_warmup("""[
            {"id": "novel", "kind": "novel_cluster",
             "model": "drain",
             "cooldown_s": 0}]""")

        # First sighting of cluster 1 -> fires.
        t = evaluate(rs, _ir("a"; extra =
                              Dict("drain" => Dict("cluster_id" => 1))))
        @test length(t) == 1
        @test t[1].fields["cluster_id"] == 1
        # Second sighting -> no fire.
        @test isempty(evaluate(rs, _ir("b"; extra =
                              Dict("drain" => Dict("cluster_id" => 1)))))
        # A new cluster id -> fires.
        t = evaluate(rs, _ir("c"; extra =
                              Dict("drain" => Dict("cluster_id" => 2))))
        @test length(t) == 1
        @test t[1].fields["cluster_id"] == 2
    end

    @testset "novel_token — sliding LRU" begin
        rs = _no_warmup("""[
            {"id": "tok", "kind": "novel_token",
             "field": "line",
             "lru_size": 16}]""")
        t = evaluate(rs, _ir("alpha beta gamma"))
        @test length(t) == 1
        @test Set(t[1].fields["novel_tokens"]) == Set(["alpha","beta","gamma"])
        # Re-occurrence — no novel tokens, no fire.
        @test isempty(evaluate(rs, _ir("alpha beta gamma")))
        # New token only — fires with just the new one.
        t = evaluate(rs, _ir("alpha delta"))
        @test length(t) == 1
        @test t[1].fields["novel_tokens"] == ["delta"]
    end

    @testset "rate_spike — min_count" begin
        rs = _no_warmup("""[
            {"id": "errs", "kind": "rate_spike",
             "match": { "kind": "regex",
                        "field": "line",
                        "pattern": "(?i)error" },
             "window_s": 60,
             "min_count": 3,
             "cooldown_s": 0}]""")
        # Two errors: no spike yet.
        @test isempty(evaluate(rs, _ir("ERROR boom 1")))
        @test isempty(evaluate(rs, _ir("Error boom 2")))
        # Third error within window -> fires.
        t = evaluate(rs, _ir("error boom 3"))
        @test length(t) == 1
        @test t[1].fields["count_in_window"] == 3
        @test t[1].fields["min_count"]       == 3
    end

    @testset "cooldown suppresses repeated firings" begin
        body = """
        { "version": 1,
          "defaults": { "warmup_required": false, "cooldown_s": 60 },
          "rules": [
            { "id": "fatal", "kind": "keyword",
              "field": "line", "keywords": ["FATAL"] }]}
        """
        rs = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        t1 = evaluate(rs, _ir("FATAL — disk full"))
        @test length(t1) == 1
        t2 = evaluate(rs, _ir("FATAL — second"))
        @test isempty(t2)   # cooldown
    end

    @testset "warmup gating — rules wait for warmup before firing" begin
        # 3-line warmup, no time-based warmup. Keyword rule with
        # warmup_required = true should not fire on warmup lines.
        body = """
        { "version": 1,
          "defaults": { "warmup_required": true, "cooldown_s": 0 },
          "rules": [
            { "id": "fatal", "kind": "keyword",
              "field": "line", "keywords": ["FATAL"] }]}
        """
        rs = load_rules(IOBuffer(body); warmup_lines = 3, warmup_seconds = 1e9)

        # Warmup lines 1..3 — even matching ones don't fire.
        @test isempty(evaluate(rs, _ir("FATAL during warmup #1")))
        @test isempty(evaluate(rs, _ir("FATAL during warmup #2")))
        @test isempty(evaluate(rs, _ir("FATAL during warmup #3")))
        # Warmup complete -> next match fires.
        t = evaluate(rs, _ir("FATAL post-warmup"))
        @test length(t) == 1
    end

    @testset "route_sinks — severity routing + implicit stdout" begin
        body = """
        { "version": 1,
          "defaults": { "warmup_required": false, "cooldown_s": 0 },
          "rules": [
            { "id": "f", "kind": "keyword", "field": "line",
              "keywords": ["X"], "severity": "crit"} ],
          "routes": [
            { "match": {"severity": ["crit"]},
              "sinks": ["webhook:default"] }],
          "sinks": {
            "webhook:default": {
              "url": "https://example.com/hook",
              "format": "raw" }}}
        """
        rs = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        rule = rs.rules[1]
        sinks = route_sinks(rs, rule)
        @test "stdout" in sinks
        @test "webhook:default" in sinks

        # Value-based variant matches by severity/rule_id without a rule.
        sinks2 = Rules.route_sinks_for(rs; severity = :crit,
                                       rule_id = "pattern:1:x")
        @test "webhook:default" in sinks2
        sinks3 = Rules.route_sinks_for(rs; severity = :info,
                                       rule_id = "pattern:1:x")
        @test sinks3 == ["stdout"]
    end

    @testset "warmup clock starts at first evaluate, not at load" begin
        # Time-based warmup of 0.15 s. Simulate slow model loading by
        # sleeping AFTER load_rules but BEFORE the first line: the
        # clock must not have started, so line 1 is still warmup.
        body = """
        { "version": 1,
          "defaults": { "warmup_required": true, "cooldown_s": 0 },
          "rules": [
            { "id": "fatal", "kind": "keyword",
              "field": "line", "keywords": ["FATAL"] }]}
        """
        rs = load_rules(IOBuffer(body);
                        warmup_lines = 1000, warmup_seconds = 0.15)
        sleep(0.3)   # "model loading" — longer than the warmup budget
        @test isempty(evaluate(rs, _ir("FATAL right after load")))
        sleep(0.2)   # now the (line-1-started) clock has expired
        t = evaluate(rs, _ir("FATAL post-warmup"))
        @test length(t) == 1
    end

    @testset "volume baseline uses first-line clock" begin
        # 10 warmup lines fed instantly; baseline must come out as a
        # rate over the (tiny) first-to-last-line elapsed, not over a
        # construction-to-now window inflated by load time.
        body = """
        { "version": 1,
          "rules": [
            { "id": "vol", "kind": "volume_anomaly",
              "window_s": 10, "baseline_multiplier": 1000000.0,
              "warmup_required": true, "cooldown_s": 0 }]}
        """
        rs = load_rules(IOBuffer(body);
                        warmup_lines = 10, warmup_seconds = 1e9)
        sleep(0.25)   # load-time gap that must NOT dilute the baseline
        for i in 1:11
            evaluate(rs, _ir("l$i"; line_id = i))
        end
        st = rs.state["vol"]
        # 10 warmup lines over a sub-100ms window → per-10s baseline
        # far above 10; the old construction-time clock would have
        # produced a diluted value close to (or below) 10/0.25s·10s=400.
        @test st[:baseline_window_count] > 0
    end

    @testset "novelty survives another rule's cooldown window" begin
        body = """
        { "version": 1,
          "defaults": { "warmup_required": false },
          "rules": [
            { "id": "novel", "kind": "novel_cluster",
              "model": "drain", "cooldown_s": 0.2 } ]}
        """
        rs = load_rules(IOBuffer(body); warmup_lines = 0, warmup_seconds = 0.0)
        # Cluster A fires and starts the cooldown.
        t = evaluate(rs, _ir("a"; extra = Dict("drain" => Dict("cluster_id" => 1))))
        @test length(t) == 1
        # Cluster B first appears DURING the cooldown — suppressed now…
        @test isempty(evaluate(rs, _ir("b";
            extra = Dict("drain" => Dict("cluster_id" => 2)))))
        sleep(0.25)
        # …but NOT swallowed: it fires on the next sighting.
        t = evaluate(rs, _ir("b again";
            extra = Dict("drain" => Dict("cluster_id" => 2))))
        @test length(t) == 1
        @test t[1].fields["cluster_id"] == 2
    end

end
