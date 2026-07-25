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
using ..Memory.PatternCatalog: PatternCatalog, pin_from_trigger, enable!, list
using Dates: Dates, DateTime, now, UTC
import REPL

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
# Latest triggers in the window (newest first), including the drain
# cluster id so the interactive actions can pin / disable by template.
function _latest_triggers(db, since::AbstractString, n::Integer = 12)
    since_ms = epoch_ms_since(String(since))
    return _rows(db, """
        SELECT id, ts, rule_id, severity, line_id, line, drain_cluster_id
        FROM triggers
        WHERE ts_epoch_ms >= ?
        ORDER BY ts_epoch_ms DESC LIMIT ?
        """, (Int(since_ms), Int(n)))
end

function render(db;
                since::AbstractString = "24h",
                limit::Integer = 12,
                latest_n::Integer = 12,
                colour::Bool = true,
                sel::Integer = 0)
    since_ms = epoch_ms_since(String(since))
    rules = top_rules_window(db; since = since_ms, limit = Int(limit))
    latest = _latest_triggers(db, since, Int(latest_n))
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
        for (i, r) in enumerate(latest)
            ts = String(r.ts)
            sev = String(r.severity)
            sev_col = sev == "crit" ? RED : sev == "warn" ? YEL : GRN
            line = String(r.line)
            # Character-safe truncation — a byte-index SubString throws
            # StringIndexError when byte 80 lands mid-UTF-8-sequence.
            line = length(line) > 80 ? first(line, 80) * "…" : line
            marker = (i == sel) ? "> " : "  "
            colour && i == sel && print(io, BOLD)
            print(io, marker, ts[12:min(end, 19)], "  ")
            colour && print(io, sev_col)
            print(io, rpad(sev, 4))
            colour && print(io, RESET)
            colour && i == sel && print(io, BOLD)
            print(io, "  ", rpad(String(r.rule_id), 22), "  ", line)
            colour && i == sel && print(io, RESET)
            println(io)
        end
    end

    println(io)
    _dim(io, "  [q]uit  [r]eload  [j/k] move  [p]in  [a]nnotate  " *
             "[d]isable-cluster", colour)
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

# ---------------------------------------------------------------------------
# Interactive actions (DB-backed, unit-tested). The key loop wires these
# to keystrokes; keeping the side effects here makes them testable
# without a TTY.
# ---------------------------------------------------------------------------

"""
    _tui_pin_trigger!(db, trigger_id) -> Int

Pin the trigger as a curated pattern: by drain template when the trigger
carries a `drain_cluster_id`, else by keywords lifted from its line.
Returns the new pattern id.
"""
function _tui_pin_trigger!(db, trigger_id::Integer)
    trow = _rows(db, "SELECT drain_cluster_id FROM triggers WHERE id = ?",
                 (Int(trigger_id),))
    isempty(trow) && throw(ArgumentError("trigger $trigger_id not found"))
    has_drain = !(trow[1].drain_cluster_id isa Missing) &&
                trow[1].drain_cluster_id !== nothing
    return pin_from_trigger(db, Int(trigger_id);
        name = "tui-pin-$(trigger_id)-$(_stamp())",
        match_kind = has_drain ? :drain : :keyword,
        created_by = "tui")
end

"""
    _tui_annotate!(db, trigger_id, note; author = "tui") -> Int

Attach a free-text `note` to a trigger (the `annotations` table).
Returns the annotation row id.
"""
function _tui_annotate!(db, trigger_id::Integer, note::AbstractString;
                        author::AbstractString = "tui")
    isempty(strip(note)) && throw(ArgumentError("annotation note is empty"))
    SQLite.DBInterface.execute(db,
        "INSERT INTO annotations (target_kind, target_id, note, author, ts) " *
        "VALUES ('trigger', ?, ?, ?, ?)",
        (Int(trigger_id), String(note), String(author), _iso_now())) |>
        (c -> foreach(identity, c))
    return Int(SQLite.last_insert_rowid(db))
end

"""
    _tui_disable_cluster!(db, cluster_id) -> Int

Disable every enabled pattern that matches drain template `cluster_id`
(stop alerting on that log template). Returns how many were disabled.
"""
function _tui_disable_cluster!(db, cluster_id::Integer)
    n = 0
    for p in list(db; enabled_only = true)
        if p.match_kind === :drain && p.match_drain_template_id == Int(cluster_id)
            enable!(db, p.id, false)
            n += 1
        end
    end
    return n
end

_stamp() = Dates.format(now(UTC), Dates.dateformat"yyyymmddHHMMSS")
_iso_now() = string(Dates.format(now(UTC),
                                 Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

"""
    run(db; since = "24h", refresh_s = 1.0)

Block in a paint loop, refreshing every `refresh_s` seconds. Non-TTY
stdout (pipes / tests) renders once and returns.

On a TTY the terminal is put in raw mode for single-key control:
`q` quit, `r` reload, `j`/`k` move the selection through the latest
triggers, `p` pin the selected trigger as a pattern, `a` annotate it
(type the note, Enter to save), `d` disable every pattern on the
selected trigger's drain cluster. Ctrl-C always exits.
"""
function run(db;
             since::AbstractString = "24h",
             refresh_s::Real = 1.0)
    if !(stdout isa Base.TTY)
        print(stdout, render(db; since = since, colour = false))
        return 0
    end
    term = REPL.Terminals.TTYTerminal(get(ENV, "TERM", "xterm"),
                                       stdin, stdout, stderr)
    keys = Channel{Char}(64)
    reader = @async begin
        try
            while true
                put!(keys, read(stdin, Char))
            end
        catch
        end
    end
    raw_ok = try; REPL.Terminals.raw!(term, true); true; catch; false; end
    sel = 1
    status = ""
    try
        while true
            latest = _latest_triggers(db, since)
            sel = clamp(sel, 1, max(1, length(latest)))
            print(stdout, "\e[2J\e[H")
            print(stdout, render(db; since = since, colour = true, sel = sel))
            isempty(status) || print(stdout, "  ", DIM, status, RESET, "\n")
            flush(stdout)
            status = ""
            deadline = time() + Float64(refresh_s)
            while time() < deadline
                if isready(keys)
                    c = take!(keys)
                    if c == 'q' || c == '\x03'          # q or Ctrl-C
                        return 0
                    elseif c == 'j'; sel += 1; break
                    elseif c == 'k'; sel -= 1; break
                    elseif c == 'r'; break
                    elseif c in ('p', 'a', 'd') && !isempty(latest)
                        t = latest[clamp(sel, 1, length(latest))]
                        status = _handle_action(db, c, t, keys)
                        break
                    end
                else
                    sleep(0.02)
                end
            end
        end
    catch e
        e isa InterruptException || rethrow()
    finally
        raw_ok && (try; REPL.Terminals.raw!(term, false); catch; end)
        try; close(keys); catch; end
    end
    return 0
end

# Dispatch a single action key against the selected trigger row `t`,
# returning a one-line status message. `a` reads a note from the same
# key channel (chars until Enter) so there's one stdin reader.
function _handle_action(db, key::Char, t, keys::Channel{Char})
    try
        if key == 'p'
            pid = _tui_pin_trigger!(db, Int(t.id))
            return "pinned trigger #$(t.id) as pattern #$pid"
        elseif key == 'd'
            if t.drain_cluster_id isa Missing || t.drain_cluster_id === nothing
                return "trigger #$(t.id) has no drain cluster to disable"
            end
            n = _tui_disable_cluster!(db, Int(t.drain_cluster_id))
            return "disabled $n pattern(s) on cluster #$(t.drain_cluster_id)"
        elseif key == 'a'
            print(stdout, "\n  note> "); flush(stdout)
            note = _read_line_from(keys)
            isempty(strip(note)) && return "annotation cancelled"
            aid = _tui_annotate!(db, Int(t.id), note)
            return "annotated trigger #$(t.id) (note #$aid)"
        end
    catch e
        return "action failed: " * sprint(showerror, e)
    end
    return ""
end

# Accumulate characters from the key channel until Enter, echoing them.
function _read_line_from(keys::Channel{Char})
    buf = IOBuffer()
    while true
        c = take!(keys)
        if c == '\r' || c == '\n'
            break
        elseif c == '\x7f' || c == '\b'                 # backspace
            s = String(take!(buf)); isempty(s) || (s = s[1:prevind(s, end)])
            print(stdout, "\r  note> ", s, "\e[K"); flush(stdout)
            print(buf, s)
        else
            print(buf, c); print(stdout, c); flush(stdout)
        end
    end
    return String(take!(buf))
end

end # module TUI
