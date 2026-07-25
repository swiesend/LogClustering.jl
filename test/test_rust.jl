using Test
using LogClustering
using LogClustering.Rust
using LogClustering.Rust: Span

# Skip the entire file gracefully when the Rust library was not built
# (e.g. `cargo` missing at `Pkg.build` time on the developer's machine).
# When a library IS present but its ABI doesn't match, fail loudly and
# DON'T run the layout-dependent FFI tests below — a stale .so with a
# different struct layout would risk a segfault, not a clean failure.
if Rust.abi_version() == 0
    @warn "logclustering_rs not built; skipping Rust tests. \
           Install Rust and `Pkg.build(\"LogClustering\")` to enable."
    @testset "Rust (skipped — library not built)" begin
        @test_skip Rust.abi_version() == Rust.ABI_VERSION
    end
elseif Rust.abi_version() != Rust.ABI_VERSION
    @error "logclustering_rs ABI mismatch (got $(Rust.abi_version()), expected $(Rust.ABI_VERSION)) — rebuild with Pkg.build; skipping FFI tests to avoid a stale-layout crash."
    @testset "Rust FFI (skipped — ABI mismatch)" begin
        @test Rust.abi_version() == Rust.ABI_VERSION   # fails loudly, no ccalls
    end
else

@testset "Rust FFI" begin
    @testset "ABI version" begin
        @test Rust.abi_version() == Rust.ABI_VERSION
    end

    @testset "parse_line (thesis Algorithm 3.1)" begin
        labels = ["ts", "ip"]
        patterns = [
            raw"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            raw"\d{1,3}(?:\.\d{1,3}){3}",
        ]
        line = "2024-01-01T00:00:00Z DEBUG 127.0.0.1 hello"
        spans = Rust.parse_line(line, labels, patterns)
        @test length(spans) == 4
        @test spans[1].label == "ts"
        @test line[spans[1].start:spans[1].stop] == "2024-01-01T00:00:00Z"
        @test spans[2].label === nothing
        @test line[spans[2].start:spans[2].stop] == " DEBUG "
        @test spans[3].label == "ip"
        @test line[spans[3].start:spans[3].stop] == "127.0.0.1"
        @test spans[4].label === nothing
        @test line[spans[4].start:spans[4].stop] == " hello"
    end

    @testset "parse_line — no match returns whole line raw" begin
        spans = Rust.parse_line("plain text", ["x"], ["DOES_NOT_MATCH"])
        @test length(spans) == 1
        @test spans[1].label === nothing
        @test spans[1].start == 1
        @test spans[1].stop == length("plain text")
    end

    @testset "parse_line — empty line" begin
        spans = Rust.parse_line("", String[], String[])
        @test isempty(spans)
    end

    @testset "parse_line — argument validation" begin
        @test_throws ArgumentError Rust.parse_line("x", ["a"], ["a", "b"])
    end

    @testset "parse_line — invalid regex raises" begin
        # Unbalanced parenthesis should cause the Rust side to return null.
        @test_throws ErrorException Rust.parse_line("x", ["bad"], ["("])
    end

    @testset "infer_regex — thesis Beispiel 3.1" begin
        s1 = ["%RCE_DATETIME%", " ", "DEBUG", " ", "-", " ", "de", ".",
              "rcenvironment", ".", "core", ".", "communication", ".",
              "transport", ".", "jms", ".", "activemq"]
        s2 = ["%RCE_DATETIME%", " ", "DEBUG", " ", "-", " ", "de", ".",
              "rcenvironment", ".", "core", ".", "communication", ".",
              "transport", ".", "jms", ".", "common"]
        out = Rust.infer_regex([s1, s2]; replacements = ["%"])
        @test out == raw".*? DEBUG \- de\.rcenvironment\.core\.communication\.transport\.jms\.(activemq|common)"
    end

    @testset "infer_regex — identical samples collapse to literals" begin
        s = ["foo", " ", "bar"]
        @test Rust.infer_regex([s, s]) == raw"foo bar"
    end

    @testset "infer_regex — disagreement produces alternation" begin
        a = ["a", "x"]
        b = ["a", "y"]
        c = ["b", "x"]
        @test Rust.infer_regex([a, b, c]) == raw"(a|b)(x|y)"
    end

    @testset "infer_regex — custom wildcard" begin
        s1 = ["%TIME%", " ", "x"]
        s2 = ["%TIME%", " ", "y"]
        out = Rust.infer_regex([s1, s2]; replacements = ["%"], wildcard = raw"\S+")
        @test out == raw"\S+ (x|y)"
    end

    @testset "infer_regex — anti-unification on variable-length samples" begin
        # Two samples of different length share a prefix and a suffix;
        # the variable middle collapses to wildcards while the anchor
        # words survive as literals. Exact whitespace placement in the
        # emit depends on LCS tie-breaking, so we assert structure
        # rather than byte-equality.
        a = ["ERROR", " ", "at", " ", "a", " ", "line", " ", "42"]
        b = ["ERROR", " ", "at", " ", "b", " ", "extra", " ", "frame",
             " ", "line", " ", "99"]
        out = Rust.infer_regex([a, b]; align = true)
        @test occursin("ERROR", out) && occursin("at", out) && occursin("line", out)
        @test occursin(".*?", out)

        # Three samples of varying length, same anchor words.
        s1 = ["user", " ", "login", " ", "ok"]
        s2 = ["user", " ", "login", " ", "failed"]
        s3 = ["user", " ", "tried", " ", "login", " ", "failed"]
        out = Rust.infer_regex([s1, s2, s3]; align = true)
        @test occursin("user", out)
        @test occursin("login", out)

        # Position-aligned mode still works on equal-length samples —
        # anti-unification is opt-in.
        @test Rust.infer_regex([["a", "b"], ["a", "b"]]; align = false) == "ab"
        @test Rust.infer_regex([["a", "b"], ["a", "b"]]; align = true)  == "ab"
    end

    @testset "parse_lines — batched API matches the per-line loop" begin
        labels   = ["TS", "IP"]
        patterns = [raw"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                    raw"\d{1,3}(?:\.\d{1,3}){3}"]
        lines = [
            "2024-01-01T00:00:00Z DEBUG 127.0.0.1 hello",
            "plain text line",
            "2024-02-02T00:00:00Z 10.0.0.5 again",
            "",
        ]
        batched = Rust.parse_lines(lines, labels, patterns)
        @test length(batched) == length(lines)
        for (i, l) in enumerate(lines)
            pl = Rust.parse_line(l, labels, patterns)
            @test length(batched[i]) == length(pl)
            for (a, b) in zip(batched[i], pl)
                @test (a.start, a.stop, a.label) == (b.start, b.stop, b.label)
            end
        end
    end

    @testset "parse_lines — empty input vector" begin
        @test Rust.parse_lines(String[], String[], String[]) == Vector{Vector{Span}}()
    end

    @testset "parse_lines — argument validation + invalid regex" begin
        @test_throws ArgumentError Rust.parse_lines(["x"], ["a"], ["a", "b"])
        @test_throws ErrorException Rust.parse_lines(["x"], ["bad"], ["("])
    end

    @testset "parse_lines — 2 k lines is input-length-scaled" begin
        labels   = ["IP"]
        patterns = [raw"\d{1,3}(?:\.\d{1,3}){3}"]
        lines = [string("line ", i, " from 10.0.0.", i % 250) for i in 1:2000]
        spans = Rust.parse_lines(lines, labels, patterns)
        @test length(spans) == 2000
        # Every line has one IP match → at least one `IP` span.
        @test all(any(s.label == "IP" for s in sp) for sp in spans)
    end

    @testset "End-to-end: parse_line feeds infer_regex" begin
        # Parse two similar lines into (timestamp label, raw tail) pairs
        # and hand the tokens to infer_regex. The label name is UPPERCASE
        # so the thesis's `%[0-9A-Z_]*?%` placeholder regex rewrites it
        # to the wildcard `.*?` per Algorithm 3.10.
        labels = ["TS"]
        patterns = [raw"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"]
        line_a = "2024-01-01T00:00:00Z DEBUG activemq"
        line_b = "2024-01-01T00:00:00Z DEBUG common"

        function tokens(line)
            out = String[]
            for s in Rust.parse_line(line, labels, patterns)
                push!(out, s.label === nothing ? line[s.start:s.stop] : "%$(s.label)%")
            end
            return out
        end

        out = Rust.infer_regex([tokens(line_a), tokens(line_b)];
                               replacements = ["%"])
        @test out == raw".*?( DEBUG activemq| DEBUG common)"
    end

    # -----------------------------------------------------------------
    # Stage E″ MDL ladder bindings
    # -----------------------------------------------------------------

    @testset "slot_ladder — exact literal" begin
        @test Rust.slot_ladder(["foo", "foo", "foo"]) == "foo"
        @test Rust.slot_ladder(["a.b", "a.b"]) == raw"a\.b"   # metachar escaped
    end

    @testset "slot_ladder — enum under enum_max" begin
        @test Rust.slot_ladder(["GET", "POST", "PUT"]; enum_max = 8) ==
              "(?:GET|POST|PUT)"
    end

    @testset "slot_ladder — typed battery wins past enum_max" begin
        typed = ["IP" => raw"\d{1,3}(?:\.\d{1,3}){3}",
                 "INT" => raw"\d+"]
        ips = ["10.0.0.$i" for i in 1:10]
        @test Rust.slot_ladder(ips; typed = typed, enum_max = 4) ==
              raw"(?:\d{1,3}(?:\.\d{1,3}){3})"
    end

    @testset "slot_ladder — bounded digit class fallback" begin
        vals = [string(i) for i in 1000:1020]
        @test Rust.slot_ladder(vals; enum_max = 4) == raw"\d{4}"
    end

    @testset "slot_ladder — wildcard on mixed shapes" begin
        @test Rust.slot_ladder(["hello", "42", "!@#"]; enum_max = 2) == ".*?"
    end

    @testset "slot_ladder — empty input → wildcard" begin
        @test Rust.slot_ladder(String[]; wildcard = "<*>") == "<*>"
    end

    @testset "alt_min — dedup + sort + escape" begin
        @test Rust.alt_min(["c", "a", "b", "a"]) == "(?:a|b|c)"
        @test Rust.alt_min(["only"]) == "only"
        @test Rust.alt_min(["x.y", "x.y"]) == raw"x\.y"
        @test Rust.alt_min(String[]; wildcard = "<*>") == "<*>"
    end

    @testset "verify_pattern — hit / total" begin
        hits, total = Rust.verify_pattern(raw"\d{1,3}(?:\.\d{1,3}){3}",
                                          ["10.0.0.1", "10.0.0.2", "not ip"])
        @test (hits, total) == (2, 3)
    end

    @testset "verify_pattern — malformed is total miss" begin
        @test Rust.verify_pattern("[broken", ["whatever"]) == (0, 1)
    end

    @testset "mdl_cost — tighter is cheaper" begin
        samples = ["hello world", "hello world"]
        tight = Rust.mdl_cost("hello world", samples)
        loose = Rust.mdl_cost(".*?", samples)
        @test tight < loose
        @test tight > 0
    end

    @testset "RegexSet — compile + match" begin
        set = Rust.RegexSet([raw"^\d+$", raw"^[a-z]+$", raw"^[A-Z]+$"])
        @test Rust.regexset_match(set, "42") == [1]
        @test Rust.regexset_match(set, "hello") == [2]
        @test Rust.regexset_match(set, "HELLO") == [3]
        @test isempty(Rust.regexset_match(set, "Mixed1"))
    end

    @testset "RegexSet — rejects malformed pattern" begin
        @test_throws ErrorException Rust.RegexSet(["[broken"])
    end

    @testset "RegexSet — stores original pattern strings" begin
        set = Rust.RegexSet([raw"^a$", raw"^b$"])
        @test set.patterns == [raw"^a$", raw"^b$"]
    end
end

end # if library built
