"""
    CLI

Shell-facing entrypoint for the package — plan 002 Stage C. One
`main(args)` function dispatches over subcommands; the shell shim
at `bin/logcluster` invokes it. Every subcommand is exposed as a
Julia-callable function too, so `CLI.train(...)` etc. drive the
same paths without going through the string-based arg parser.

Subcommands:

- `train`       — fit a model. `--kind deep_kate | vq_vae | seq_lstm
                  | drain`, `--data FILE`, `--out FILE`, optional
                  `--auto` + `--budget N`, `--epochs N`.
- `classify`    — load a saved parser / model and emit
                  `(line, cluster_id, template)` per input line.
- `score`       — anomaly score per line, fusing the template
                  reconstruction with value-novelty when a saved
                  detector is supplied.
- `benchmark`   — thin wrapper around `benchmarks/loghub2/run.jl`.
- `mask`        — apply the typed-slot regex battery to stdin /
                  `--data`.
- `download-loghub` — fetch the 2 k subsets.

The CLI deliberately **avoids** Comonicon's `@main` macro so the
subcommand table lives in plain Julia data — easier to test in-
process and to extend. Argument parsing uses the stdlib + a tiny
long-flag matcher; if a dependency on Comonicon is desirable later
for PackageCompiler binary generation, the move is mechanical.
"""
module CLI

using ..Harness: load_loghub, run_parser, format_report, Dataset
using ..Masking: mask_line, mask_lines, mask_lines_with_values
using ..Drain3: Drain, process!, parse_all
using ..Persistence
using ..PersistenceGlue
using ..AutoTune
using ..DeepKATE: deep_kate
using ..VQVAE: vq_vae, assign_codes
using ..SeqLSTM: seq_lstm
using ..Instance: ValueNoveltyDetector, update!, anomaly_score, combined_anomaly
using ..Framing: parse_frame, SOURCE_RAW
using JSON3

export main

# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------

"""
    main(args::AbstractVector{<:AbstractString} = ARGS) -> Int

CLI entry. Returns a process exit code so shell shims can
`exit(CLI.main(ARGS))` and callers can `@test CLI.main([...]) == 0`
without shelling out. Writes errors to stderr; subcommand output
goes to stdout or `--out`.
"""
function main(args::AbstractVector{<:AbstractString} = ARGS)::Int
    if isempty(args) || args[1] in ("-h", "--help")
        _print_top_help()
        return 0
    end
    cmd = args[1]
    rest = @view args[2:end]
    handler = get(SUBCOMMANDS, cmd, nothing)
    handler === nothing && return _bad_command(cmd)
    try
        return handler(collect(String, rest))
    catch err
        if err isa ArgumentError || err isa ErrorException
            println(stderr, "logcluster $cmd: ", err.msg)
            return 2
        end
        rethrow()
    end
end

function _bad_command(cmd)
    println(stderr, "logcluster: unknown command `$cmd`. Try `--help`.")
    return 2
end

function _print_top_help()
    println("""
    usage: logcluster <command> [args...]

    commands:
      train            fit a model on a log corpus
      classify         label each line with its template / cluster id
      score            per-line anomaly score
      mask             apply the typed-slot regex battery
      benchmark        run one parser vs a LogHub-2.0 CSV
      download-loghub  fetch the 2k subsets

    Run `logcluster <command> --help` for subcommand flags.
    """)
end

# ---------------------------------------------------------------------------
# Argument parsing — minimal, self-contained.
# ---------------------------------------------------------------------------

"""
    parse_flags(args, specs) -> Dict{String, Any}

Tiny `--flag value` / `--flag=value` / `--bool` parser. `specs` is a
Vector of `(name, default, kind)` where `kind ∈ (:string, :int,
:float, :bool, :path)`. Positional arguments aren't supported —
every subcommand takes everything as `--flag`.
"""
function parse_flags(args::AbstractVector{<:AbstractString},
                     specs::AbstractVector{<:Tuple})
    out = Dict{String, Any}()
    for (name, default, _) in specs
        out[name] = default
    end
    kinds = Dict{String, Symbol}(String(n) => k for (n, _, k) in specs)
    i = 1
    while i <= length(args)
        tok = args[i]
        if tok in ("-h", "--help")
            out["help"] = true
            i += 1
            continue
        end
        startswith(tok, "--") ||
            throw(ArgumentError("unexpected positional arg `$tok`; every flag starts with --"))
        eq = findfirst('=', tok)
        name, val = if eq === nothing
            tok[3:end], nothing
        else
            tok[3:prevind(tok, eq)], tok[nextind(tok, eq):end]
        end
        haskey(kinds, name) ||
            throw(ArgumentError("unknown flag `--$name`"))
        kind = kinds[name]
        if kind === :bool
            out[name] = val === nothing ? true : lowercase(val) in ("1", "true", "yes")
            i += 1
        else
            if val === nothing
                i + 1 > length(args) &&
                    throw(ArgumentError("--$name needs a value"))
                val = args[i + 1]
                i += 2
            else
                i += 1
            end
            out[name] = _coerce(val, kind)
        end
    end
    return out
end

function _coerce(val::AbstractString, kind::Symbol)
    kind === :string && return String(val)
    kind === :path   && return String(val)
    kind === :int    && return parse(Int, val)
    kind === :float  && return parse(Float64, val)
    throw(ArgumentError("unknown flag kind `$kind`"))
end

# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------

"Read lines from `path`, or all of stdin when `path == \"-\"` or empty."
function read_lines(path::AbstractString)
    if isempty(path) || path == "-"
        return readlines(stdin)
    end
    return readlines(path)
end

"Write a text block to `path` (or stdout when empty/-)."
function write_out(path::AbstractString, block::AbstractString)
    if isempty(path) || path == "-"
        print(block)
    else
        open(path, "w") do io
            print(io, block)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------

function cmd_mask(args::Vector{String})::Int
    specs = [
        ("data",     "-",       :path),
        ("out",      "-",       :path),
        ("template", "<\$LABEL>", :string),
        ("framed",   false,     :bool),
        ("values",   false,     :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_mask_help(); return 0)
    lines = read_lines(opts["data"])
    bodies = opts["framed"] ? [String(parse_frame(l).message) for l in lines] : lines

    if opts["values"]
        tpls, vals = mask_lines_with_values(bodies; template = opts["template"])
        io = IOBuffer()
        for (t, vs) in zip(tpls, vals)
            obj = Dict{String, Any}("template" => t,
                                    "values" => [Dict("label" => v.label,
                                                      "value" => v.value,
                                                      "start" => v.start,
                                                      "stop"  => v.stop) for v in vs])
            println(io, JSON3.write(obj))
        end
        write_out(opts["out"], String(take!(io)))
    else
        tpls = mask_lines(bodies; template = opts["template"])
        write_out(opts["out"], join(tpls, '\n') * (isempty(tpls) ? "" : "\n"))
    end
    return 0
end

function _print_mask_help()
    println("""
    usage: logcluster mask [--data FILE] [--out FILE] [--template TPL]
                           [--framed] [--values]

    Apply the default typed-slot regex battery to stdin / --data.
    --template controls the output form (default `<\$LABEL>`).
    --framed    strips the collector envelope first (RFC 5424 / CRI / Docker).
    --values    emits one JSON object per line with `template` + `values[]`
                (label, value, 1-based byte range).
    """)
end

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

function cmd_train(args::Vector{String})::Int
    specs = [
        ("kind",   "drain",  :string),
        ("data",   "-",      :path),
        ("out",    "",       :path),
        ("auto",   false,    :bool),
        ("budget", 0,        :int),
        ("epochs", 1,        :int),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_train_help(); return 0)
    isempty(opts["out"]) && throw(ArgumentError("--out is required"))
    lines = read_lines(opts["data"])
    kind = Symbol(opts["kind"])
    if kind === :drain
        d = Drain()
        parse_all(d, lines)
        PersistenceGlue.save(opts["out"], d;
                             metadata = Dict{String, Any}(
                                 "source" => opts["data"],
                                 "n_lines" => length(lines),
                                 "kind"   => "drain",
                             ))
        println(stderr,
                "drain: ", length(d.clusters),
                " templates from ", length(lines), " lines → ", opts["out"])
        return 0
    elseif kind === :deep_kate || kind === :vq_vae || kind === :seq_lstm
        println(stderr, """
            `train --kind $kind` needs a featurised corpus (numeric
            Matrix or tokenised sequence), which the CLI doesn't
            featurise from raw lines yet. Pick `--kind drain` for
            raw lines, or use the Julia API directly:

                using LogClustering
                cfg = AutoTune.fit_hyperparams($(repr(kind)), X)
                m = $(kind == :deep_kate ? "deep_kate" :
                      kind == :vq_vae ? "vq_vae" : "seq_lstm")(cfg.n; …)
                # train… save with PersistenceGlue.save(path, m, ps, st; kind, …)
            """)
        return 3
    else
        throw(ArgumentError("unknown kind `$(opts["kind"])`"))
    end
end

function _print_train_help()
    println("""
    usage: logcluster train --kind KIND --data FILE --out FILE
                            [--auto] [--budget N] [--epochs N]

    Fit and persist a model.

    KIND:
      drain       streaming log-template parser (no featurisation needed)
      deep_kate   Lux autoencoder (requires a feature matrix — API-only)
      vq_vae      Lux codebook AE (requires a feature matrix — API-only)
      seq_lstm    Lux next-event LSTM (requires tokenised sequences — API-only)

    --auto        use AutoTune.fit_hyperparams to pick sizes from the data.
    --budget N    >0 triggers random search around the heuristic seed.
    --epochs N    training epochs (model-specific).
    """)
end

# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

function cmd_classify(args::Vector{String})::Int
    specs = [
        ("model",  "",    :path),
        ("data",   "-",   :path),
        ("out",    "-",   :path),
        ("format", "tsv", :string),
        ("framed", false, :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_classify_help(); return 0)
    isempty(opts["model"]) && throw(ArgumentError("--model is required"))
    artifact = Persistence.load_and_rehydrate(opts["model"])

    lines = read_lines(opts["data"])
    bodies = opts["framed"] ? [String(parse_frame(l).message) for l in lines] : lines

    if artifact isa Drain
        ids = Int[]
        templates = String[]
        for l in bodies
            cid, tpl = process!(artifact, l)
            push!(ids, cid); push!(templates, tpl)
        end
        _emit_classify(opts["out"], opts["format"], bodies, ids, templates)
        return 0
    else
        throw(ArgumentError("model kind $(typeof(artifact)) not yet handled by classify"))
    end
end

function _emit_classify(out_path, fmt, lines, ids, templates)
    io = IOBuffer()
    if fmt == "tsv"
        println(io, "line_id\tcluster_id\ttemplate\tline")
        for i in eachindex(lines)
            println(io, i, '\t', ids[i], '\t',
                    replace(templates[i], '\t' => ' '), '\t',
                    replace(lines[i], '\t' => ' '))
        end
    elseif fmt == "json"
        for i in eachindex(lines)
            println(io, JSON3.write(Dict(
                "line_id"    => i,
                "cluster_id" => ids[i],
                "template"   => templates[i],
                "line"       => lines[i],
            )))
        end
    else
        throw(ArgumentError("unknown --format `$fmt`; use tsv or json"))
    end
    write_out(out_path, String(take!(io)))
end

function _print_classify_help()
    println("""
    usage: logcluster classify --model PATH [--data FILE] [--out FILE]
                               [--format tsv|json] [--framed]

    Load a saved model (currently: Drain) and emit one record per
    input line with its cluster id and inferred template.
    """)
end

# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

function cmd_score(args::Vector{String})::Int
    specs = [
        ("detector", "", :path),
        ("data",     "-", :path),
        ("out",      "-", :path),
        ("format",   "tsv", :string),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_score_help(); return 0)
    isempty(opts["detector"]) && throw(ArgumentError("--detector is required"))
    det = Persistence.load_and_rehydrate(opts["detector"])
    det isa ValueNoveltyDetector ||
        throw(ArgumentError("--detector must be a ValueNoveltyDetector bundle"))
    lines = read_lines(opts["data"])
    _, values = mask_lines_with_values(lines)
    scores = Float64[]
    for vs in values
        push!(scores, Float64(LogClustering.Instance.value_novelty(det, vs)))
    end
    io = IOBuffer()
    if opts["format"] == "tsv"
        println(io, "line_id\tscore\tline")
        for i in eachindex(lines)
            println(io, i, '\t', scores[i], '\t', replace(lines[i], '\t' => ' '))
        end
    elseif opts["format"] == "json"
        for i in eachindex(lines)
            println(io, JSON3.write(Dict(
                "line_id" => i,
                "score"   => scores[i],
                "line"    => lines[i],
            )))
        end
    else
        throw(ArgumentError("unknown --format `$(opts["format"])`"))
    end
    write_out(opts["out"], String(take!(io)))
    return 0
end

function _print_score_help()
    println("""
    usage: logcluster score --detector PATH [--data FILE] [--out FILE]
                            [--format tsv|json]

    Load a saved ValueNoveltyDetector, mask each line, and emit one
    anomaly score per line. A template-reconstruction head can be
    added once DeepKATE/VQVAE featurisation lands.
    """)
end

# ---------------------------------------------------------------------------
# benchmark / download-loghub — thin wrappers that delegate to the existing
# scripts under `benchmarks/loghub2/`.
# ---------------------------------------------------------------------------

function cmd_benchmark(args::Vector{String})::Int
    # Lazy-include to avoid hard-wiring the path at CLI module load time.
    here = @__DIR__
    run_jl = joinpath(here, "..", "benchmarks", "loghub2", "run.jl")
    isfile(run_jl) ||
        throw(ArgumentError("benchmark script not found at $run_jl"))
    # `run.jl` reads `ARGS` directly; hand it our tail and call its main.
    mod = Module(:LcBenchmarkShim)
    Core.eval(mod, :(ARGS = $(copy(args))))
    Core.eval(mod, :(include($(run_jl))))
    Core.eval(mod, :(main(ARGS)))
    return 0
end

function cmd_download(args::Vector{String})::Int
    here = @__DIR__
    dl_jl = joinpath(here, "..", "benchmarks", "loghub2", "download.jl")
    isfile(dl_jl) ||
        throw(ArgumentError("download script not found at $dl_jl"))
    mod = Module(:LcDownloadShim)
    Core.eval(mod, :(ARGS = $(copy(args))))
    Core.eval(mod, :(include($(dl_jl))))
    Core.eval(mod, :(main(ARGS)))
    return 0
end

# ---------------------------------------------------------------------------
# Subcommand table.
# ---------------------------------------------------------------------------

const SUBCOMMANDS = Dict{String, Function}(
    "train"            => cmd_train,
    "classify"         => cmd_classify,
    "score"            => cmd_score,
    "mask"             => cmd_mask,
    "benchmark"        => cmd_benchmark,
    "download-loghub"  => cmd_download,
)

end # module CLI
