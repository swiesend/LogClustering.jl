#
# End-to-end smoke test of the ported thesis pipeline.
#
#   julia --project examples/smoke.jl
#
# Walks six synthetic log lines through the ported modules in order:
#
#   Framing → parse_line (Rust, Alg. 3.1)
#          → infer_regex  (Rust, Alg. 3.10)
#          → DeepKATE     (Lux, §3.2.3 + §3.2.5)
#          → Episodes     (Julia, §3.2.7)
#
# Prints one titled section per stage; no assertions — if anything
# raises, the script exits non-zero.

using LogClustering
using LogClustering.Framing
using LogClustering.Rust
using LogClustering.DeepKATE
using LogClustering.Instance
using LogClustering.Episodes
using Lux
using Random

const LINES = [
    "<165>1 2024-04-17T10:00:00Z app-1 evntslog 1 ID47 [src@1 ip=\"10.0.0.1\"] user=alice login ok",
    "<165>1 2024-04-17T10:00:01Z app-1 evntslog 1 ID48 [src@1 ip=\"10.0.0.2\"] user=bob   login ok",
    "<165>1 2024-04-17T10:00:02Z app-1 evntslog 1 ID49 [src@1 ip=\"10.0.0.3\"] user=carol login failed",
    "2024-04-17T10:00:03.000Z stdout F POST /api/v1/login 200 23ms",
    "2024-04-17T10:00:04.000Z stderr F POST /api/v1/login 500 120ms",
    "{\"log\":\"level=info msg=healthcheck\\n\",\"stream\":\"stdout\",\"time\":\"2024-04-17T10:00:05Z\"}",
]

hdr(title) = (println(); println(repeat("─", 72)); println("  ", title); println(repeat("─", 72)))

hdr("1. Framing")
frames = [parse_frame(l) for l in LINES]
for (i, f) in enumerate(frames)
    println(lpad(i, 2), "  ", rpad(string(f.source), 22),
            "  ts=", rpad(f.timestamp, 28),
            "  msg=", repr(String(f.message)))
end

hdr("2. parse_line on message bodies (Rust, Alg. 3.1)")
labels   = ["TS", "IP", "HTTP_STATUS", "DURATION_MS", "NUM"]
patterns = [
    raw"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
    raw"\d{1,3}(?:\.\d{1,3}){3}",
    raw"\b[1-5]\d\d\b",
    raw"\b\d+ms\b",
    raw"\b\d+\b",
]

"""Turn a span decomposition into tokens, substituting `%LABEL%` for matches."""
function tokenise(line, spans, labels)
    [s.label === nothing ? line[s.start:s.stop] : "%$(s.label)%" for s in spans]
end

tokens_per_line = Vector{String}[]
for (i, f) in enumerate(frames)
    body = String(f.message)
    isempty(body) && continue
    spans = Rust.parse_line(body, labels, patterns)
    toks = tokenise(body, spans, labels)
    push!(tokens_per_line, toks)
    println(lpad(i, 2), "  ", body)
    for s in spans
        tag = s.label === nothing ? "raw" : s.label
        println("       ", rpad(tag, 14), repr(body[s.start:s.stop]))
    end
end

hdr("3. infer_regex across the 3 syslog lines (Rust, Alg. 3.10)")
# The three syslog message bodies share a common template; feed them as a
# cluster through the thesis's position-aligned inference.
syslog_tokens = tokens_per_line[1:3]
rx = Rust.infer_regex(syslog_tokens; replacements = ["%"])
println("  ", rx)

hdr("4. DeepKATE forward + per-sample reconstruction error (Lux, §3.2.3 + §3.2.5)")
# Tiny synthetic bag-of-tokens: per-line vectors of token-position frequencies,
# normalised to [0, 1]. Enough to exercise the forward and the Anomaly scoring.
rng = Random.MersenneTwister(17)
vocab = unique(reduce(vcat, tokens_per_line))
vocab_idx = Dict(t => i for (i, t) in enumerate(vocab))
n = length(vocab)

X = zeros(Float32, n, length(tokens_per_line))
for (j, toks) in enumerate(tokens_per_line)
    for t in toks
        X[vocab_idx[t], j] += 1f0
    end
end
X ./= maximum(X; dims = 1) .+ 1f-6     # per-line max-normalise

model = deep_kate(n; latent = 2, k1 = 4)
ps, st = Lux.setup(rng, model)
scores = anomaly_score(model, ps, st, X;
                       weights = (abs = 1.0, sq = 1.0, latent = 0.0),
                       normalise = true)
println(lpad("line", 4), "  ", lpad("tokens", 6), "  ", lpad("score", 8), "  body")
for (i, (s, toks)) in enumerate(zip(scores, tokens_per_line))
    body = String(frames[i].message)
    println(lpad(i, 4), "  ", lpad(length(toks), 6), "  ",
            lpad(round(s; digits = 4), 8), "  ", body)
end
println("(`score` ∈ [0,1], higher = less well reconstructed; \
random untrained weights here, so ranking is illustrative only.)")

hdr("5. MV-Span on the line-id sequence (Julia, §3.2.7)")
# Treat the 6 lines as an event-type sequence. The syslog lines get id 1,
# CRI lines id 2, Docker id 3, raw id 4 (not present here).
event_ids = map(frames) do f
    f.source === Framing.SOURCE_SYSLOG_5424  ? 1 :
    f.source === Framing.SOURCE_K8S_CRI      ? 2 :
    f.source === Framing.SOURCE_DOCKER_JSONL ? 3 : 4
end
# Repeat the run so MV-Span has something to discover.
seq = repeat(event_ids, 3)
println("  event-id sequence: ", seq)
db = mv_span(seq; min_sup = 2, max_gap = -1)
println("  episodes discovered (pattern → #occurrences):")
for (k, v) in sort(collect(db); by = kv -> -length(kv[1]))
    println("    ", k, "  →  ", length(v))
end

println()
println("✔ smoke pipeline completed.")
