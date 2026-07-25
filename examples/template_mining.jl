#
# Template mining over an Apache-like access log.
#
#   julia --project examples/template_mining.jl
#
#   parse_line     (Rust, Alg. 3.1)  — typed-slot decomposition
#   → cluster by label-sequence + raw literals  (structural skeleton)
#   → infer_regex  (Rust, Alg. 3.10) — regex per cluster
#   → verify       (Julia Regex)     — each regex matches its own cluster
#                                      and rejects the others.

using LogClustering
using LogClustering.Rust

const LINES = [
    "10.0.0.1 alice [17/Apr/2026:10:00:00 +0000] GET /api/v1/users 200 1423 12ms",
    "10.0.0.2 bob [17/Apr/2026:10:00:01 +0000] GET /api/v1/users 200 1390 9ms",
    "10.0.0.3 carol [17/Apr/2026:10:00:02 +0000] GET /api/v1/users 200 1511 15ms",
    "10.0.0.4 dave [17/Apr/2026:10:00:03 +0000] POST /api/v1/login 200 42 23ms",
    "10.0.0.5 eve [17/Apr/2026:10:00:04 +0000] POST /api/v1/login 401 17 31ms",
    "10.0.0.6 frank [17/Apr/2026:10:00:05 +0000] POST /api/v1/login 200 41 19ms",
    "10.0.0.7 grace [17/Apr/2026:10:00:06 +0000] DELETE /api/v1/sessions/abc 204 0 8ms",
    "10.0.0.8 heidi [17/Apr/2026:10:00:07 +0000] DELETE /api/v1/sessions/def 204 0 7ms",
    "10.0.0.9 ivan [17/Apr/2026:10:00:08 +0000] GET /metrics 200 8401 2ms",
    "10.0.0.1 judy [17/Apr/2026:10:00:09 +0000] GET /metrics 200 8355 2ms",
    "10.0.0.2 karl [17/Apr/2026:10:00:10 +0000] POST /api/v1/login 500 0 240ms",
    "10.0.0.3 liam [17/Apr/2026:10:00:11 +0000] GET /api/v1/users 200 1402 11ms",
]

# Rank order matters (complex → simple). METHOD before USER so HTTP
# verbs don't get swallowed by the lowercase-ident rule; DUR_MS before
# BYTES so `12ms` isn't decoded as plain digits.
const LABELS = ["IP", "DATE", "METHOD", "STATUS", "DUR_MS", "BYTES", "USER"]
const PATTERNS = [
    raw"\d{1,3}(?:\.\d{1,3}){3}",
    raw"\[\d{1,2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\]",
    raw"\b(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b",
    raw"\b[1-5]\d\d\b",
    raw"\b\d+ms\b",
    raw"\b\d+\b",
    raw"\b[a-z]+\b",
]
@assert length(LABELS) == length(PATTERNS)

hdr(title) = (println(); println(repeat("─", 76)); println("  ", title); println(repeat("─", 76)))

function tokenise(line, spans)
    [s.label === nothing ? line[s.start:s.stop] : "%$(s.label)%" for s in spans]
end

"Collapse internal whitespace runs to a single space; preserve label tokens."
function normalise(tokens)
    out = String[]
    for t in tokens
        if startswith(t, "%")
            push!(out, t)
        else
            # Collapse any run of whitespace to one space. Emit the token
            # only if it still contains content (including a lone space).
            norm = replace(t, r"\s+" => " ")
            isempty(norm) || push!(out, norm)
        end
    end
    return out
end

function main()
    parsed = [Rust.parse_line(l, LABELS, PATTERNS) for l in LINES]
    tokens = [normalise(tokenise(l, s)) for (l, s) in zip(LINES, parsed)]

    hdr("1. Parsed spans (line 1)")
    println("    ", LINES[1])
    for s in parsed[1]
        tag = s.label === nothing ? "raw" : s.label
        println("      ", rpad(tag, 8), " ", repr(LINES[1][s.start:s.stop]))
    end

    # Skeleton = the full normalised token sequence. Two lines share a
    # skeleton iff they have the same label sequence AND the same
    # literal runs between labels. Tokens are joined directly because
    # they already carry any necessary whitespace.
    skeletons = [join(t) for t in tokens]
    clusters  = Dict{String, Vector{Int}}()
    for (i, sk) in enumerate(skeletons)
        push!(get!(clusters, sk, Int[]), i)
    end
    cluster_ids = sort!(collect(keys(clusters)); by = k -> -length(clusters[k]))

    hdr("2. Skeleton-level clusters")
    for (ci, sk) in enumerate(cluster_ids)
        mem = clusters[sk]
        println("  C", ci, " — ", length(mem), " line(s): ", mem)
        println("      ", sk)
    end

    hdr("3. Inferred regex per cluster — typed-class ladder (Stage E″)")
    # The class map tightens the per-label wildcard to the original
    # typed regex — `%IP%` → `\d{1,3}(?:\.\d{1,3}){3}` etc. — instead
    # of letting everything collapse to `.*?`.
    classes = Dict{String, String}(LABELS[i] => PATTERNS[i] for i in 1:length(LABELS))
    cluster_regexes = Regex[]
    for (ci, sk) in enumerate(cluster_ids)
        idxs = clusters[sk]
        rx_str = Rust.infer_regex(tokens[idxs]; replacements = ["%"], classes = classes)
        push!(cluster_regexes, Regex("^" * rx_str * "\$"))
        println("  C", ci, "  ", rx_str)
    end

    hdr("4. Verification grid (• match, × reject)")
    print(lpad("line", 10))
    for ci in 1:length(cluster_regexes)
        print(lpad("C$ci", 4))
    end
    println()

    # Verify against the original source lines (whitespace-normalised so
    # the multi-space runs in LINES[2,3,…] don't trip an otherwise
    # correct regex). The regex is anchored to the whole line.
    haystacks = [replace(l, r"\s+" => " ") for l in LINES]
    cells = 0
    hits  = 0
    for (li, hay) in enumerate(haystacks)
        print("  line ", lpad(li, 2), "  ")
        for (ci, rx) in enumerate(cluster_regexes)
            matched = occursin(rx, hay)
            expected = li in clusters[cluster_ids[ci]]
            print(lpad(matched ? "•" : "×", 4))
            cells += 1
            hits  += matched == expected ? 1 : 0
        end
        println()
    end
    println()
    println("  ", hits, "/", cells, " cells correct  (",
            round(100hits/cells; digits = 1), "%)")
end

main()
