"""
    Memory.Redis

Minimal RESP-2 client (TCP + Sockets, no new deps) backing the
[`WarmStore`] interface. Implements just the commands the rule
engine needs:

  - `SADD` / `SISMEMBER` / `SMEMBERS`   (novel-cluster sets)
  - `ZADD` / `ZREMRANGEBYSCORE` / `ZCARD` (sliding windows)
  - `SET` / `GET` / `EXPIRE` / `DEL`     (cooldown, reservoirs)
  - `PING`                               (health check)
  - `FLUSHDB` (tests only — use AUTH-gated DBs!)

RESP-2 framing is enough for our purposes; RESP-3 brings new types
we don't use. Connection is single-shot per process; reconnect on
EOF.

## Surface

- [`Client`] — connection state.
- [`connect`] / [`close`] — lifecycle.
- [`call`] — generic command dispatch returning the parsed reply.
- thin wrappers (`sadd!`, `sismember`, `zadd!`, …) used by
  [`WarmStore`].
"""
module RedisClient

using Sockets

export Client, connect_redis, sadd!, sismember, smembers, zadd!,
       zremrangebyscore!, zcard, set!, get_str, expire!, del!,
       ping, flushdb!, call

# ---------------------------------------------------------------------------
# Connection.
# ---------------------------------------------------------------------------

mutable struct Client
    host::String
    port::Int
    db::Int
    password::String
    sock::Union{Nothing, Sockets.TCPSocket}
    Client(host, port, db, password) = new(host, port, db, password, nothing)
end

"""
    connect_redis(url::AbstractString) -> Client

Parse a `redis://[:password@]host[:port][/db]` URL and return a
ready-to-use client (the socket is opened lazily on first command).
"""
function connect_redis(url::AbstractString)
    host, port, db, pw = _parse_url(url)
    c = Client(host, port, db, pw)
    _open!(c)
    return c
end

function Base.close(c::Client)
    if c.sock !== nothing
        try; close(c.sock); catch; end
        c.sock = nothing
    end
    return nothing
end

function _parse_url(url::AbstractString)
    s = String(url)
    startswith(s, "redis://") ||
        throw(ArgumentError("redis URL must start with redis://, got `$s`"))
    rest = s[length("redis://") + 1 : end]
    pw = ""
    at = findlast('@', rest)
    if at !== nothing
        cred = rest[1:prevind(rest, at)]
        rest = rest[nextind(rest, at):end]
        # Form is `user:password` or `:password` — keep only the password.
        colon = findfirst(':', cred)
        pw = colon === nothing ? cred : cred[nextind(cred, colon):end]
    end
    host = rest; port = 6379; db = 0
    slash = findfirst('/', rest)
    if slash !== nothing
        host = rest[1:prevind(rest, slash)]
        dbstr = rest[nextind(rest, slash):end]
        db = isempty(dbstr) ? 0 : parse(Int, dbstr)   # trailing "/" ⇒ db 0
    end
    # IPv6 literal: `[::1]` or `[::1]:6379`. Strip brackets and split the
    # port on the ']' boundary so a colon inside the address isn't
    # mistaken for the port separator.
    if startswith(host, "[")
        close_br = findfirst(']', host)
        close_br === nothing && throw(ArgumentError("malformed IPv6 redis host: $host"))
        addr = host[2:prevind(host, close_br)]
        after = host[nextind(host, close_br):end]
        if startswith(after, ":")
            port = parse(Int, after[2:end])
        end
        host = addr
    else
        colon = findlast(':', host)
        if colon !== nothing
            port = parse(Int, host[nextind(host, colon):end])
            host = host[1:prevind(host, colon)]
        end
    end
    return host, port, db, pw
end

function _open!(c::Client)
    c.sock = Sockets.connect(c.host, c.port)
    if !isempty(c.password)
        call(c, "AUTH", c.password) == "OK" ||
            error("Redis AUTH failed")
    end
    if c.db != 0
        call(c, "SELECT", string(c.db)) == "OK" ||
            error("Redis SELECT $(c.db) failed")
    end
    return c
end

function _ensure_open!(c::Client)
    if c.sock === nothing || !isopen(c.sock)
        _open!(c)
    end
    return c.sock
end

# ---------------------------------------------------------------------------
# RESP-2 protocol.
# ---------------------------------------------------------------------------

"""
    call(c::Client, args::AbstractString...) -> Any

Send one Redis command and return the parsed reply. Bulk strings
come back as `String`; integers as `Int`; arrays as `Vector{Any}`;
`nil` replies as `nothing`. Errors raise `ErrorException`.

A half-dead connection (server restarted, NAT idle-timeout) usually
still reports `isopen`, so the write/read here fails rather than the
`isopen` check in `_ensure_open!`. On an I/O failure the socket is
closed and the command retried **once** on a fresh connection; a
second failure propagates. Redis command errors (`-ERR ...`) are
NOT retried — they are not connection problems.
"""
function call(c::Client, args::AbstractString...)
    try
        sock = _ensure_open!(c)
        _write_command(sock, args)
        return _read_reply(sock)
    catch e
        # A RESP command error carries "Redis: " — surfaced, not a
        # transport failure, so don't reconnect.
        (e isa ErrorException && occursin("Redis: ", e.msg)) && rethrow()
        # Transport failure — drop the socket and retry once.
        try; c.sock === nothing || close(c.sock); catch; end
        c.sock = nothing
        sock = _ensure_open!(c)
        _write_command(sock, args)
        return _read_reply(sock)
    end
end

call(c::Client, args::Vector{String}) = call(c, args...)

function _write_command(sock, args)
    n = length(args)
    buf = IOBuffer()
    write(buf, "*", string(n), "\r\n")
    for a in args
        s = String(a)
        write(buf, "\$", string(sizeof(s)), "\r\n")
        write(buf, s)
        write(buf, "\r\n")
    end
    write(sock, take!(buf))
    return nothing
end

function _read_reply(sock)
    line = _read_line(sock)
    isempty(line) && error("Redis: empty reply")
    prefix = line[1]
    body = line[2:end]
    if prefix == '+'
        return body
    elseif prefix == '-'
        error("Redis: $body")
    elseif prefix == ':'
        return parse(Int, body)
    elseif prefix == '$'
        n = parse(Int, body)
        n == -1 && return nothing
        data = read(sock, n)
        # Consume the trailing CRLF.
        read(sock, 2)
        return String(data)
    elseif prefix == '*'
        n = parse(Int, body)
        n == -1 && return nothing
        out = Vector{Any}(undef, n)
        for i in 1:n
            out[i] = _read_reply(sock)
        end
        return out
    else
        error("Redis: unknown reply prefix `$prefix` in $line")
    end
end

function _read_line(sock)
    buf = IOBuffer()
    while !eof(sock)
        b = read(sock, UInt8)
        if b == UInt8('\r')
            # Consume the LF.
            read(sock, 1)
            break
        end
        write(buf, b)
    end
    return String(take!(buf))
end

# ---------------------------------------------------------------------------
# Command wrappers (the subset the WarmStore needs).
# ---------------------------------------------------------------------------

# SADD key member [member ...] -> Int (count of newly added)
function sadd!(c::Client, key::AbstractString, members::Vector{<:AbstractString})
    isempty(members) && return 0
    return Int(call(c, "SADD", key, members...))
end
sadd!(c::Client, key::AbstractString, m::AbstractString) = sadd!(c, key, [m])

# SISMEMBER key value -> 0 | 1
function sismember(c::Client, key::AbstractString, value::AbstractString)
    return Int(call(c, "SISMEMBER", key, value)) == 1
end

# SMEMBERS key -> Vector{String}
function smembers(c::Client, key::AbstractString)
    r = call(c, "SMEMBERS", key)
    r === nothing && return String[]
    return String[String(x) for x in r]
end

# ZADD key score member -> Int
function zadd!(c::Client, key::AbstractString, score::Real, member::AbstractString)
    return Int(call(c, "ZADD", key, string(Float64(score)), member))
end

# ZREMRANGEBYSCORE key min max -> Int
function zremrangebyscore!(c::Client, key::AbstractString,
                            min::Real, max::Real)
    return Int(call(c, "ZREMRANGEBYSCORE", key,
                    string(Float64(min)), string(Float64(max))))
end

# ZCARD key -> Int
function zcard(c::Client, key::AbstractString)
    return Int(call(c, "ZCARD", key))
end

# SET key value [EX seconds] -> "OK"
function set!(c::Client, key::AbstractString, value::AbstractString;
              ex::Union{Nothing, Integer} = nothing)
    args = String["SET", String(key), String(value)]
    if ex !== nothing
        push!(args, "EX"); push!(args, string(Int(ex)))
    end
    return call(c, args)
end

# GET key -> String | nothing
function get_str(c::Client, key::AbstractString)
    r = call(c, "GET", key)
    return r === nothing ? nothing : String(r)
end

# EXPIRE key seconds -> 0 | 1
function expire!(c::Client, key::AbstractString, seconds::Integer)
    return Int(call(c, "EXPIRE", key, string(Int(seconds)))) == 1
end

# DEL key [key ...] -> Int
function del!(c::Client, keys::Vector{<:AbstractString})
    isempty(keys) && return 0
    return Int(call(c, "DEL", keys...))
end
del!(c::Client, key::AbstractString) = del!(c, [key])

function ping(c::Client)
    return call(c, "PING") == "PONG"
end

function flushdb!(c::Client)
    return call(c, "FLUSHDB") == "OK"
end

end # module RedisClient
