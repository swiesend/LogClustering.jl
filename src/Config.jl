"""
    Config

TOML-backed configuration loader that merges three layers, in
priority order (highest wins):

  1. CLI flags (whatever `parse_flags` parses)
  2. `~/.config/logcluster/config.toml` (or `\$LOGCLUSTERING_CONFIG`)
  3. Bundled defaults baked into the subcommand's `specs` block.

Each subcommand has its own `[stream]` / `[insights]` / etc. block;
unknown keys are tolerated so a future-version config file doesn't
crash an older binary.

## Surface

- [`default_path`] — resolved config-file location.
- [`load`] — parse the TOML into a nested Dict.
- [`merge_into!`] — overlay a config section onto a `parse_flags`
  output dict, leaving flag-provided values untouched.
"""
module Config

using TOML

export default_path, load, merge_into!

"""
    default_path() -> String

Resolved config-file location, honouring `\$LOGCLUSTERING_CONFIG`
when set, else `\$XDG_CONFIG_HOME/logcluster/config.toml`, else
`~/.config/logcluster/config.toml`.
"""
function default_path()
    haskey(ENV, "LOGCLUSTERING_CONFIG") && return ENV["LOGCLUSTERING_CONFIG"]
    cfg = get(ENV, "XDG_CONFIG_HOME",
              joinpath(try; homedir(); catch; "."; end, ".config"))
    return joinpath(cfg, "logcluster", "config.toml")
end

"""
    load(path::AbstractString = default_path()) -> Dict{String, Any}

Parse the config file. Returns an empty `Dict` when the file
doesn't exist. Raises `ArgumentError` on malformed TOML.
"""
function load(path::AbstractString = default_path())
    isfile(path) || return Dict{String, Any}()
    try
        return TOML.parsefile(path)
    catch e
        throw(ArgumentError("config: failed to parse $path: $(sprint(showerror, e))"))
    end
end

"""
    merge_into!(opts::AbstractDict, section::AbstractDict;
                cli_provided::AbstractSet)

Overlay every key in `section` onto `opts` unless the operator
explicitly supplied it on the command line (i.e. the key is in
`cli_provided`). Returns `opts`.

The CLI's `parse_flags` doesn't currently track provenance, so the
overlay logic uses a simpler rule: a key is "CLI-provided" when its
value differs from the spec default. Callers compute that set once
and pass it in.
"""
function merge_into!(opts::AbstractDict, section::AbstractDict;
                     cli_provided::AbstractSet = Set{String}())
    for (k, v) in section
        ks = String(k)
        ks in cli_provided && continue
        # Skip keys the subcommand doesn't know about — keeps a
        # future-version config file from crashing today's binary.
        haskey(opts, ks) || continue
        opts[ks] = v
    end
    return opts
end

end # module Config
