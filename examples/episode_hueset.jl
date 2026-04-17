#
# High-utility episode mining with MT-Span.
#
#   julia --project examples/episode_hueset.jl
#
# Synthetic security-event sequence: two rare but profitable workflows
# ("exploit kill-chain" and "data exfiltration") are embedded in a noisy
# background of common events. Each event type has an external profit.
# MT-Span is asked for episodes whose *relative* utility clears
# `min_utility = 0.4` under `max_time_duration = 6`.

using LogClustering
using LogClustering.Episodes
using LogClustering.Episodes: total_utility, external_utility

hdr(t) = (println(); println(repeat("─", 68)); println("  ", t); println(repeat("─", 68)))

# ─── event vocabulary ───────────────────────────────────────────────────
# 1..4  background noise (heartbeat, healthcheck, info log, sync)
# 5..8  kill-chain (recon → foothold → escalation → persistence)
# 9..11 exfiltration (stage → encrypt → upload)
const NAMES = [
    "heartbeat", "healthcheck", "info", "sync",
    "recon", "foothold", "escalation", "persistence",
    "stage", "encrypt", "upload",
]

const UTIL = Dict{Int, Float64}(
    # noise events cost a little
    1 => 0.1, 2 => 0.1, 3 => 0.2, 4 => 0.3,
    # kill-chain is high-value
    5 => 4.0, 6 => 5.0, 7 => 6.0, 8 => 7.0,
    # exfiltration even higher (final act)
    9 => 5.0, 10 => 7.0, 11 => 10.0,
)

# ─── synthetic trace ────────────────────────────────────────────────────
# Background: long random drizzle of 1..4. Inject kill-chain twice and
# the exfiltration chain once.
function build_trace()
    rng = 0                                    # deterministic; no RNG needed
    out = Int[]
    function noise!(n)
        for _ in 1:n
            push!(out, 1 + (length(out) * 7) % 4)   # cheap PRNG on length
        end
    end
    noise!(12)
    append!(out, [5, 6, 7, 8])                 # kill-chain #1
    noise!(10)
    append!(out, [5, 6, 8])                    # partial chain (missing 7)
    noise!(6)
    append!(out, [9, 10, 11])                  # exfiltration
    noise!(8)
    append!(out, [5, 6, 7, 8])                 # kill-chain #2
    noise!(6)
    return out
end

seq = build_trace()
hdr("1. Sequence")
println("  length: ", length(seq))
println("  symbols: ", collect(zip(seq, [NAMES[i] for i in seq])))

# ─── total utility and singleton view ───────────────────────────────────
tu = total_utility(seq, UTIL)
hdr("2. Totals")
println("  total utility = ", round(tu; digits = 2),
        "  (sum of profits across the whole trace)")
for e in 1:length(NAMES)
    cnt = count(==(e), seq)
    cnt == 0 && continue
    base = external_utility(UTIL, tu, [e])
    println("    ", rpad(NAMES[e], 12), "  count=", lpad(cnt, 2),
            "  u(e)=", lpad(round(UTIL[e]; digits = 1), 5),
            "  rel=", round(base; digits = 3))
end

# ─── MT-Span ────────────────────────────────────────────────────────────
hdr("3. mv_span(min_sup=1, max_gap=2, max_time_duration=6) + post-rank by utility")
# MV-Span's per-prefix utility filter (§3.2.7) would delete the
# attack-chain *prefixes* before their multi-event extensions are ever
# explored — each single kill-chain event has rel-U well below 10 %.
# In practice the thesis recommends mining structurally first and
# re-ranking by utility afterwards. That's what we do here.
db = mv_span(seq;
             min_sup = 1,
             max_gap = 2,
             max_time_duration = 6)

# Post-rank all 3+-event episodes by external relative utility.
multi = [(pat, occs) for (pat, occs) in db if length(pat) >= 3]
sort!(multi;
      by = kv -> (-external_utility(UTIL, tu, kv[1]), -length(kv[1])))

println(lpad("rel-U", 8), "  len  support  episode")
for (pat, occs) in multi[1:min(10, end)]
    u = external_utility(UTIL, tu, pat)
    u < 0.10 && break
    name = join((NAMES[e] for e in pat), " → ")
    println(lpad(round(u; digits = 3), 8), "   ",
            lpad(length(pat), 2), "      ",
            lpad(length(occs), 2), "     ", name)
end

println()
println("  Total patterns mined         : ", length(db))
println("  Multi-event (len ≥ 3)        : ", length(multi))
println("  With rel-U ≥ 0.10 (shown)    : ",
        count(kv -> external_utility(UTIL, tu, kv[1]) >= 0.10, multi))
