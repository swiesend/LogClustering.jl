"""
    Stream

Streaming subcommand scaffolding: a bounded-channel pipeline that
reads log lines from stdin or a tailed file (with rotation safety),
hands them to a worker for inference + rule evaluation, and emits
JSON-line trigger records on stdout. This file currently houses the
**producer** half (the tailer and the line-event type); the
inference + sink stages will be added in subsequent commits.

The producer never blocks on a slow consumer: it pushes onto a
`Channel{LineEvent}` whose capacity is bounded by the caller. Lines
longer than `max_line_bytes` are silently dropped and counted on the
returned `TailStats`. Rotation (inode change or truncation) is
detected on every read-loop iteration so `logrotate create`-mode
deployments keep working.

## Surface

- [`LineEvent`] — a single line + line_id + ingestion timestamp.
- [`TailStats`] — counters for the status heartbeat.
- [`stream_stdin!`] — push lines from an `IO` onto a channel.
- [`tail_file!`] — long-running tailer with rotation + truncation
  handling.
- [`StopSignal`] — cooperative shutdown handle; producers poll it.
"""
module Stream

using Dates: Dates, DateTime, now, UTC
using ..StructuredLog: StructuredLog

export LineEvent, TailStats, StopSignal, stream_stdin!, tail_file!,
       stop!, is_stopped

# ---------------------------------------------------------------------------
# Types.
# ---------------------------------------------------------------------------

"""
    LineEvent(line, line_id, ts; partial = false)

One log line wrapped with monotonic id and ingest timestamp.
`partial = true` marks a line flushed because the line-stale timeout
fired before a trailing newline arrived.
"""
struct LineEvent
    line::String
    line_id::Int
    ts::DateTime
    partial::Bool
end

LineEvent(line::AbstractString, line_id::Integer, ts::DateTime; partial::Bool = false) =
    LineEvent(String(line), Int(line_id), ts, partial)

"""
    TailStats()

Mutable counters mirrored into the streaming status heartbeat.
"""
Base.@kwdef mutable struct TailStats
    lines_total::Int       = 0
    bytes_total::Int       = 0
    dropped_oversize::Int  = 0
    rotations::Int         = 0
    partial_flushes::Int   = 0
end

"""
    StopSignal()

Single-shot cooperative shutdown flag. Producers poll
[`is_stopped`] and exit cleanly; [`stop!`] is invoked by the signal
handler.
"""
mutable struct StopSignal
    stopped::Bool
    StopSignal() = new(false)
end

@inline is_stopped(s::StopSignal) = s.stopped
@inline stop!(s::StopSignal)      = (s.stopped = true; nothing)

# ---------------------------------------------------------------------------
# Stdin / IO source.
# ---------------------------------------------------------------------------

"""
    stream_stdin!(io, ch::Channel{LineEvent};
                  stop::StopSignal = StopSignal(),
                  stats::TailStats = TailStats(),
                  max_line_bytes::Int = 64 * 1024,
                  start_line_id::Int = 1) -> TailStats

Read newline-terminated records from `io` and push them onto `ch`.
Lines longer than `max_line_bytes` are dropped and counted on
`stats`. Returns the populated `stats`. The channel is **not**
closed by this function — the caller decides when to close it.
"""
function stream_stdin!(io::IO, ch::Channel{LineEvent};
                       stop::StopSignal = StopSignal(),
                       stats::TailStats = TailStats(),
                       max_line_bytes::Int = 64 * 1024,
                       start_line_id::Int = 1)
    line_id = start_line_id
    while !is_stopped(stop) && !eof(io)
        line = readline(io)
        nbytes = sizeof(line) + 1   # add the consumed newline
        stats.bytes_total += nbytes
        if sizeof(line) > max_line_bytes
            stats.dropped_oversize += 1
            StructuredLog.warn("line dropped";
                               reason = "oversize",
                               bytes  = sizeof(line),
                               limit  = max_line_bytes,
                               line_id = line_id)
            line_id += 1
            continue
        end
        stats.lines_total += 1
        put!(ch, LineEvent(line, line_id, now(UTC); partial = false))
        line_id += 1
    end
    return stats
end

# ---------------------------------------------------------------------------
# File tailer (rotation + truncation safe).
# ---------------------------------------------------------------------------

"""
    tail_file!(path, ch::Channel{LineEvent};
               stop::StopSignal = StopSignal(),
               stats::TailStats = TailStats(),
               from_start::Bool = false,
               max_line_bytes::Int = 64 * 1024,
               poll_seconds::Float64 = 0.05,
               line_stale_seconds::Float64 = 5.0,
               start_line_id::Int = 1) -> TailStats

Follow `path` forever (or until `stop!` is invoked). On EOF the
tailer sleeps `poll_seconds` and retries. Rotation is detected on
every loop iteration by `stat`-ing the path: an inode change or a
size that fell below the current read offset triggers a re-open
(`logrotate create`-mode flow).

Partial-line safety: if the file ends without a trailing newline,
the buffer is held until either a newline arrives or
`line_stale_seconds` passes, at which point the partial is flushed
with `partial = true` on the resulting `LineEvent`.

`stats.bytes_total` counts bytes read (not lines emitted), so a
truncated file's byte counter still reflects bytes consumed before
the rotation.
"""
function tail_file!(path::AbstractString, ch::Channel{LineEvent};
                    stop::StopSignal = StopSignal(),
                    stats::TailStats = TailStats(),
                    from_start::Bool = false,
                    max_line_bytes::Int = 64 * 1024,
                    poll_seconds::Float64 = 0.05,
                    line_stale_seconds::Float64 = 5.0,
                    start_line_id::Int = 1)
    line_id = start_line_id
    io = open(path, "r")
    from_start || seekend(io)
    file_inode = _inode(path)
    partial_buf = IOBuffer()
    partial_started_at = NaN

    try
        while !is_stopped(stop)
            # Detect rotation before each read pass.
            cur_inode = _inode(path)
            cur_size  = _filesize(path)
            cur_pos   = try; position(io); catch; 0; end
            if cur_inode !== nothing && cur_inode != file_inode
                close(io)
                io = open(path, "r")
                file_inode = cur_inode
                stats.rotations += 1
                StructuredLog.info("file rotated"; path = path, reason = "inode_change")
                # Reset partial buffer — its contents belonged to the old fd.
                partial_buf = IOBuffer()
                partial_started_at = NaN
                continue
            elseif cur_size !== nothing && cur_size < cur_pos
                # Truncation: same inode, but file shrunk.
                close(io)
                io = open(path, "r")
                stats.rotations += 1
                StructuredLog.info("file rotated"; path = path, reason = "truncation")
                partial_buf = IOBuffer()
                partial_started_at = NaN
                continue
            end

            # Read whatever bytes are available right now.
            n_emitted = 0
            while !eof(io) && !is_stopped(stop)
                chunk = readavailable(io)
                isempty(chunk) && break
                stats.bytes_total += length(chunk)
                # Walk the chunk byte-by-byte, splitting at newlines and
                # honouring the oversize cutoff at flush time so we don't
                # buffer a multi-MB blob forever.
                for b in chunk
                    if b == UInt8('\n')
                        line = String(take!(partial_buf))
                        partial_started_at = NaN
                        n_emitted += 1
                        if sizeof(line) > max_line_bytes
                            stats.dropped_oversize += 1
                            StructuredLog.warn("line dropped";
                                               reason = "oversize",
                                               bytes  = sizeof(line),
                                               limit  = max_line_bytes,
                                               line_id = line_id)
                            line_id += 1
                            continue
                        end
                        stats.lines_total += 1
                        put!(ch, LineEvent(line, line_id, now(UTC); partial = false))
                        line_id += 1
                    else
                        write(partial_buf, b)
                        if isnan(partial_started_at)
                            partial_started_at = time()
                        end
                    end
                end
            end

            # If a partial line has been hanging around past the stale
            # window, flush it with partial = true so downstream sees
            # something rather than blocking the worker forever.
            if !isnan(partial_started_at) && partial_buf.size > 0 &&
               (time() - partial_started_at) >= line_stale_seconds
                line = String(take!(partial_buf))
                partial_started_at = NaN
                if sizeof(line) <= max_line_bytes
                    stats.lines_total += 1
                    stats.partial_flushes += 1
                    put!(ch, LineEvent(line, line_id, now(UTC); partial = true))
                    line_id += 1
                else
                    stats.dropped_oversize += 1
                end
            end

            # No progress -> nap. If we did emit lines, loop again
            # immediately so we don't lose a chunk to a sibling write.
            if n_emitted == 0 && !is_stopped(stop)
                sleep(poll_seconds)
            end
        end
    finally
        try; close(io); catch; end
    end
    return stats
end

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

function _inode(path::AbstractString)
    try
        s = stat(path)
        return s.inode
    catch
        return nothing
    end
end

function _filesize(path::AbstractString)
    try
        return filesize(path)
    catch
        return nothing
    end
end

end # module Stream
