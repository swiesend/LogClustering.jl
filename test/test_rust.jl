using Test
using LogClustering
using LogClustering.Rust
using LogClustering.Rust: Span

# Skip the entire file gracefully when the Rust library was not built
# (e.g. `cargo` missing at `Pkg.build` time on the developer's machine).
if Rust.abi_version() == 0
    @warn "logclustering_rs not built; skipping Rust tests. \
           Install Rust and `Pkg.build(\"LogClustering\")` to enable."
    @testset "Rust (skipped — library not built)" begin
        @test_skip Rust.abi_version() == Rust.ABI_VERSION
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
end

end # if library built
