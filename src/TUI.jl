"""
    TUI

`htop`-style live dashboard against the SQLite memory store. Pure
ANSI-escape rendering — no new dependencies — and the renderer is
exposed as a string builder so tests can snapshot the output
without spinning a terminal.

## Surface

- [`render`] — build one frame as a `String`. Tests use this.
- [`run`] — clear-screen-and-paint loop with `q`-to-quit on a TTY.
"""
module TUI

using ..Memory.SQLite: SQLite, open_db, migrate!, _rows
using ..Memory.Insights: top_rules_window, novel_clusters_window
using ..Memory.SQLite: epoch_ms_since
using Dates: Dates, DateTime, now, UTC

export render, run

# ---------------------------------------------------------------------------
# ANSI helpers.
# ---------------------------------------------------------------------------

const CSI = "\e["
const RESET = CSI * "0m"
const BOLD  = CSI * "1m"
const DIM   = CSI * "2m"
const RED   = CSI * "31m"
const YEL   = CSI * "33m"
const GRN   = CSI * "32m"
const CYAN  = CSI * "36m"
const CLEAR = CSI * "2J" * CSI * "H"

# ---------------------------------------------------------------------------
# Frame rendering.
# ---------------------------------------------------------------------------

"""
    render(db; since = "24h", limit = 12, latest_n = 12, colour = true) -> String

Build one frame. With `colour = false`, ANSI escapes are stripped so
tests can match on plain text. The frame includes a one-line
header (uptime / total triggers), a "top rules" bar-chart panel,
and a "latest triggers" panel.
"""
function render(db;
                since::AbstractString = "24h",
                limit::Integer = 12,
                latest_n::Integer = 12,
                colour::Bool = true)
    since_ms = epoch_ms_since(String(since))
    rules = top_rules_window(db; since = since_ms, limit = Int(limit))
    latest = _rows(db, """
        SELECT id, ts, rule_id, severity, line_id, line
        FROM triggers
        WHERE ts_epoch_ms >= ?
        ORDER BY ts_epoch_ms DESC LIMIT ?
        """, (Int(since_ms), Int(latest_n)))
    novel = novel_clusters_window(db; since = since_ms)

    total = sum(Int(r.n) for r in rules; init = 0)
    max_n = isempty(rules) ? 1 : maximum(Int(r.n) for r in rules)

    io = IOBuffer()
    _printbar(io, " LogClustering — window $since   triggers $total   " *
                  "rules $(length(rules))   novel-clusters $(length(novel))   ",
              colour)
    println(io)
    _h2(io, "Top rules", colour); println(io)
    if isempty(rules)
        _dim(io, "  (no triggers in window)", colour); println(io)
    else
        for r in rules
            sev = String(r.severity)
            sev_col = sev == "crit" ? RED : sev == "warn" ? YEL : GRN
            bar = repeat("█", max(1, Int(round((Int(r.n) / max_n) * 40))))
            print(io, "  ", lpad(string(r.n), 5), "  ")
            colour && print(io, sev_col)
            print(io, rpad(String(r.rule_id), 32), "  ", bar)
            colour && print(io, RESET)
            println(io)
        end
    end

    println(io)
    _h2(io, "Latest triggers", colour); println(io)
    if isempty(latest)
        _dim(io, "  (no recent triggers)", colour); println(io)
    else
        for r in latest
            ts = String(r.ts)
            sev = String(r.severity)
            sev_col = sev == "crit" ? RED : sev == "warn" ? YEL : GRN
            line = String(r.line)
            line = sizeof(line) > 80 ? String(SubString(line, 1, 80)) * "…" : line
            print(io, "  ", ts[12:min(end, 19)], "  ")
            colour && print(io, sev_col)
            print(io, rpad(sev, 4))
            colour && print(io, RESET)
            print(io, "  ", rpad(String(r.rule_id), 22), "  ", line)
            println(io)
        end
    end

    println(io)
    _dim(io, "  [q] quit   [r] reload   updates every 1s", colour)
    println(io)
    return String(take!(io))
end

function _printbar(io, text::AbstractString, colour::Bool)
    if colour
        print(io, CYAN, BOLD, text, RESET)
    else
        print(io, text)
    end
end

function _h2(io, text::AbstractString, colour::Bool)
    if colour
        print(io, BOLD, text, RESET)
    else
        print(io, text)
    end
end

function _dim(io, text::AbstractString, colour::Bool)
    if colour
        print(io, DIM, text, RESET)
    else
        print(io, text)
    end
end

# ---------------------------------------------------------------------------
# Run loop.
# ---------------------------------------------------------------------------

"""
    run(db; since = "24h", refresh_s = 1.0, on_quit = nothing)

Block in a paint loop, refreshing every `refresh_s` seconds.
Non-TTY stdout (pipes / tests) renders once and returns.

A future iteration of this can register key handlers; for v1 the
operator's escape hatch is Ctrl-C, which raises InterruptException.
"""
function run(db;
             since::AbstractString = "24h",
             refresh_s::Real = 1.0)
    if !(stdout isa Base.TTY)
        print(stdout, render(db; since = since, colour = false))
        return 0
    end
    try
        while true
            print(stdout, "\e[2J\e[H")
            print(stdout, render(db; since = since, colour = true))
            flush(stdout)
            sleep(Float64(refresh_s))
        end
    catch e
        e isa InterruptException || rethrow()
    end
    return 0
end

end # module TUI
