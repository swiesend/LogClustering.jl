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
using ..Dedup: Dedup
using ..Drain3: Drain, process!, parse_all
using ..Featurise: Featurise, Vocabulary, build_vocab, bow,
                   sequence_matrix, tokenise_ids
using ..Persistence
using ..PersistenceGlue
using ..AutoTune
using ..DeepKATE: DeepKATE, deep_kate
using ..VQVAE: vq_vae, assign_codes, VectorQuantizer
using ..SeqLSTM: seq_lstm, seq_lstm_loss
using ..Transformer: Transformer, transformer_encoder, transformer_decoder,
                     transformer_decoder_loss, transformer_decoder_nll,
                     transformer_encoder_loss, embed_sequences
using ..Instance: ValueNoveltyDetector, update!, anomaly_score, combined_anomaly,
                  value_novelty
using ..Sparsity: sparsity_clusters
using ..Pipeline: Pipeline, l2_normalise, kmeans_cluster
using ..Framing: parse_frame, SOURCE_RAW
using ..RCA: RCA, root_cause, root_cause_sparsity, render_markdown
using ..Rules: Rules
using ..StructuredLog: StructuredLog
using ..Stream: Stream
using ..SinksWebhook: SinksWebhook
using ..Memory: Memory
using ..Config: Config
using ..TUI: TUI
using Lux
using Zygote
using JSON3
using Dates: Dates, DateTime, now, UTC
using Random: MersenneTwister
using Statistics: mean

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
      stream           long-running line-by-line inference + rule alerts
      rules            print / validate / explain / dry-run rule bundles
      patterns         pin / list / explain user-curated patterns (SQLite)
      query            triggers / sessions / patterns from the SQLite store
      insights         top rules / novel templates / episodes / transitions
      report           Markdown / JSON report bundling the insights
      top              live TUI dashboard against the SQLite store
      init             bootstrap config / rules / sqlite under ~/.config
      doctor           verify config / model / db / Redis connectivity
      select-model     find a registry bundle matching a corpus
      rca              root-cause-analysis report (cluster+anomaly+episodes)
      mask             apply the typed-slot regex battery
      benchmark        run one parser vs a LogHub-2.0 CSV
      download-loghub  fetch the 2k subsets

    Exit codes:
      0   success (clean shutdown signal for `stream`).
      1   at least one rule fired (only when --exit-on-trigger is set).
      2   argument / config / schema error.
      3   I/O error (model bundle missing, source file gone).
      130 killed by SIGINT / SIGTERM during shutdown.

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
    # Names the operator explicitly set on the command line, so the
    # config-file overlay (`_apply_config!`) knows which keys NOT to
    # override. Stashed under a reserved key ignored by all consumers.
    provided = Set{String}()
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
        push!(provided, name)
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
    out["__cli_provided"] = provided
    return out
end

"""
    _apply_config!(opts, section::AbstractString)

Overlay the `[section]` block of the user's config.toml onto `opts`,
skipping keys the operator set on the command line (tracked in
`opts["__cli_provided"]`) and keys the subcommand doesn't define.
Values are coerced to the type of the existing default. Precedence:
CLI flag > config file > built-in default.
"""
function _apply_config!(opts::AbstractDict, section::AbstractString)
    cfg = try
        Config.load()
    catch e
        StructuredLog.warn("config ignored"; error = sprint(showerror, e))
        return opts
    end
    haskey(cfg, section) || return opts
    provided = get(opts, "__cli_provided", Set{String}())
    for (k, v) in cfg[section]
        ks = String(k)
        (ks in provided || !haskey(opts, ks)) && continue
        cur = opts[ks]
        opts[ks] = cur isa Bool ? Bool(v) :              # Bool <: Integer — test first
                   cur isa Integer && v isa Number ? Int(v) :
                   cur isa AbstractFloat && v isa Number ? Float64(v) :
                   cur isa AbstractString ? String(v) :
                   v
    end
    return opts
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
        ("kind",      "drain",  :string),
        ("data",      "-",      :path),
        ("out",       "",       :path),
        ("auto",      false,    :bool),
        ("budget",    0,        :int),
        ("epochs",    20,       :int),
        ("batch",     64,       :int),
        ("lr",        0.05,     :float),
        ("seed",      0,        :int),
        ("seqlen",    16,       :int),
        ("max-vocab", 5000,     :int),
        ("min-count", 1,        :int),
        ("reuse",     false,    :bool),
        ("registry",  _default_registry_path(), :path),
        ("quiet",     false,    :bool),
        # Transformer-only flags. Defaults match the planning doc; CLI
        # ignores them for non-transformer kinds.
        ("d-model",    128,     :int),
        ("n-layers",   4,       :int),
        ("n-heads",    8,       :int),
        ("n-kv-heads", 0,       :int),    # 0 → default to n-heads ÷ 4 (min 1)
        ("ffn-mult",   4.0,     :float),
        ("dropout",    0.0,     :float),
        ("mask-rate",  0.15,    :float),  # encoder MLM only
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_train_help(); return 0)
    isempty(opts["out"]) && throw(ArgumentError("--out is required"))
    lines = read_lines(opts["data"])
    kind = Symbol(opts["kind"])
    rng = MersenneTwister(Int(opts["seed"]))

    # --reuse short-circuit: if the registry holds a bundle whose
    # corpus fingerprint matches the current vocab, copy it to --out
    # and skip training. Only supported for kinds that save a vocab.
    if opts["reuse"] && kind in (:deep_kate, :vq_vae, :seq_lstm,
                                 :transformer_encoder, :transformer_decoder)
        reused = _try_reuse(kind, lines, opts)
        reused == 0 && return 0
    end

    if kind === :drain
        d = Drain()
        parse_all(d, lines)
        PersistenceGlue.save(opts["out"], d;
                             metadata = _train_metadata(opts, lines, "drain"))
        opts["quiet"] || println(stderr,
                "drain: ", length(d.clusters),
                " templates from ", length(lines), " lines → ", opts["out"])
        return 0
    elseif kind === :deep_kate
        return _train_deep_kate(lines, opts, rng)
    elseif kind === :vq_vae
        return _train_vq_vae(lines, opts, rng)
    elseif kind === :seq_lstm
        return _train_seq_lstm(lines, opts, rng)
    elseif kind === :transformer_encoder
        return _train_transformer(lines, opts, rng, :encoder)
    elseif kind === :transformer_decoder
        return _train_transformer(lines, opts, rng, :decoder)
    else
        throw(ArgumentError("unknown kind `$(opts["kind"])`"))
    end
end

"""
    _default_registry_path() -> String

Per-user model cache directory; follows the XDG-cache convention on
Linux (`~/.cache/logclustering/models`), falls back to a homedir
subdirectory otherwise. The directory is created on demand.
"""
function _default_registry_path()
    home = try; homedir(); catch; "."; end
    cache = get(ENV, "XDG_CACHE_HOME", joinpath(home, ".cache"))
    return joinpath(cache, "logclustering", "models")
end

"""
    _try_reuse(kind, lines, opts) -> Int

Build the query fingerprint for the current corpus and scan the
registry for a compatible saved bundle. Returns 0 on a successful
reuse (bundle copied to `--out`), non-zero to signal "train
instead." The fingerprint is computed from the vocab the bundle
*would* use, so reuse costs one `build_vocab` call.
"""
function _try_reuse(kind::Symbol, lines::Vector{String}, opts::Dict)::Int
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    fp = Persistence.corpus_fingerprint(vocab.tokens; n_lines = length(lines))
    registry = String(opts["registry"])
    candidate = Persistence.find_compatible_bundle(registry, kind, fp)
    candidate === nothing && return 1                     # no reuse; train.
    cp(candidate.path, String(opts["out"]); force = true)
    opts["quiet"] || println(stderr,
        "reuse: copied ", candidate.path, " → ", opts["out"],
        " (fingerprint ", fp[1:12], "…)")
    return 0
end

function _train_metadata(opts, lines, kind_name::AbstractString)
    return Dict{String, Any}(
        "source"  => opts["data"],
        "n_lines" => length(lines),
        "kind"    => kind_name,
    )
end

# --- DeepKATE ---------------------------------------------------------------

function _train_deep_kate(lines::Vector{String}, opts::Dict, rng)::Int
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    X = bow(lines, vocab; normalise = :l1)
    cfg_base = AutoTune.fit_hyperparams(:deep_kate, X;
                  budget = opts["auto"] ? Int(opts["budget"]) : 0,
                  rng = rng)
    n = size(X, 1)
    cfg = merge(cfg_base, (n = n,))
    model = deep_kate(cfg.n; hidden = cfg.hidden, latent = cfg.latent,
                      k1 = cfg.k1, k_bottleneck = cfg.k_bottleneck, p = cfg.p)
    ps, st = Lux.setup(rng, model)
    ps = _sgd_recon!(model, ps, st, X, Int(opts["epochs"]), Int(opts["batch"]),
                     Float32(opts["lr"]), rng; quiet = opts["quiet"],
                     label = "deep_kate")
    PersistenceGlue.save(opts["out"], model, ps, st;
        kind = :deep_kate,
        n = cfg.n, hidden = cfg.hidden, latent = cfg.latent,
        k1 = cfg.k1, k_bottleneck = cfg.k_bottleneck, p = cfg.p,
        vocab = vocab,
        n_lines = length(lines),
        metadata = _train_metadata(opts, lines, "deep_kate"))
    opts["quiet"] || println(stderr,
        "deep_kate: ", length(vocab), "-tok vocab, hidden=", cfg.hidden,
        ", latent=", cfg.latent, ", k1=", cfg.k1, " → ", opts["out"])
    return 0
end

# --- VQ-VAE -----------------------------------------------------------------

function _train_vq_vae(lines::Vector{String}, opts::Dict, rng)::Int
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    X = bow(lines, vocab; normalise = :binary)
    cfg_base = AutoTune.fit_hyperparams(:vq_vae, X;
                  budget = opts["auto"] ? Int(opts["budget"]) : 0,
                  rng = rng)
    n = size(X, 1)
    cfg = merge(cfg_base, (n = n,))
    model = vq_vae(cfg.n; codebook_size = cfg.codebook_size,
                   embed_dim = cfg.embed_dim, hidden = cfg.hidden)
    ps, st = Lux.setup(rng, model)
    ps = _sgd_vqvae!(model, ps, st, X, Int(opts["epochs"]), Int(opts["batch"]),
                     Float32(opts["lr"]), rng; quiet = opts["quiet"])
    PersistenceGlue.save(opts["out"], model, ps, st;
        kind = :vq_vae,
        n = cfg.n, codebook_size = cfg.codebook_size,
        embed_dim = cfg.embed_dim, hidden = cfg.hidden,
        vocab = vocab,
        metadata = _train_metadata(opts, lines, "vq_vae"))
    opts["quiet"] || println(stderr,
        "vq_vae: ", length(vocab), "-tok vocab, codebook=", cfg.codebook_size,
        ", embed=", cfg.embed_dim, " → ", opts["out"])
    return 0
end

# --- SeqLSTM ----------------------------------------------------------------

function _train_seq_lstm(lines::Vector{String}, opts::Dict, rng)::Int
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    seqlen = Int(opts["seqlen"])
    seqlen >= 2 || throw(ArgumentError("--seqlen must be ≥ 2"))
    S = sequence_matrix(lines, vocab; seqlen = seqlen)
    cfg = AutoTune.fit_hyperparams(:seq_lstm, S;
                  budget = opts["auto"] ? Int(opts["budget"]) : 0,
                  rng = rng)
    # Respect the actual vocab size — AutoTune's `vocab_size` comes
    # from `maximum(corpus)`, which can undershoot when a line didn't
    # use every token.
    vs = max(Int(cfg.vocab_size), length(vocab))
    cfg = merge(cfg, (vocab_size = vs,))
    model = seq_lstm(cfg.vocab_size; embed = cfg.embed, hidden = cfg.hidden)
    ps, st = Lux.setup(rng, model)
    ps = _sgd_seqlstm!(model, ps, st, S, Int(opts["epochs"]),
                       Int(opts["batch"]), Float32(opts["lr"]), rng;
                       quiet = opts["quiet"])
    PersistenceGlue.save(opts["out"], model, ps, st;
        kind = :seq_lstm,
        vocab_size = cfg.vocab_size, embed = cfg.embed, hidden = cfg.hidden,
        vocab = vocab, seqlen = seqlen,
        metadata = _train_metadata(opts, lines, "seq_lstm"))
    opts["quiet"] || println(stderr,
        "seq_lstm: ", length(vocab), "-tok vocab, embed=", cfg.embed,
        ", hidden=", cfg.hidden, ", seqlen=", seqlen, " → ", opts["out"])
    return 0
end

# --- Transformer (encoder + decoder) ----------------------------------------
#
# Both flavours share `_train_transformer`. The `flavour ∈ (:encoder,
# :decoder)` switch picks the right factory, loss function, save kind,
# and label string. Hyperparameters come straight from the CLI flags
# (no AutoTune entry yet — the planning doc keeps that out of scope).

"""
    _resolve_transformer_dims(opts) -> NamedTuple

Snap CLI-supplied `--n-heads` / `--n-kv-heads` / `--d-model` to a
self-consistent set. Enforces the GQA invariants:

  - `d_model % n_heads == 0`
  - `n_heads % n_kv_heads == 0`
  - `head_dim = d_model ÷ n_heads` is even (RoPE constraint)
"""
function _resolve_transformer_dims(opts)
    d_model  = Int(opts["d-model"])
    n_heads  = Int(opts["n-heads"])
    n_kv_in  = Int(opts["n-kv-heads"])
    n_kv     = n_kv_in <= 0 ? max(1, n_heads ÷ 4) : n_kv_in
    d_model % n_heads == 0 ||
        throw(ArgumentError("--d-model ($d_model) must be divisible by --n-heads ($n_heads)"))
    n_heads % n_kv == 0 ||
        throw(ArgumentError("--n-heads ($n_heads) must be divisible by --n-kv-heads ($n_kv)"))
    head_dim = d_model ÷ n_heads
    iseven(head_dim) ||
        throw(ArgumentError("derived head_dim ($head_dim) must be even (RoPE); " *
                            "adjust --d-model / --n-heads"))
    return (d_model = d_model, n_heads = n_heads, n_kv_heads = n_kv,
            head_dim = head_dim)
end

function _train_transformer(lines::Vector{String}, opts::Dict, rng,
                            flavour::Symbol)::Int
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    seqlen = Int(opts["seqlen"])
    seqlen >= 2 ||
        throw(ArgumentError("--seqlen must be ≥ 2 for transformers"))
    S = sequence_matrix(lines, vocab; seqlen = seqlen)
    dims = _resolve_transformer_dims(opts)
    vocab_size = max(Int(maximum(S)), length(vocab))

    # `max_seq_len` rounds up to the next power of 2 (cheap RoPE cache,
    # leaves headroom for slightly longer inference inputs).
    max_seq_len = 1 << ceil(Int, log2(max(2, seqlen)))

    label, kind, factory, loss_fn = if flavour === :encoder
        ("tx_enc", :transformer_encoder, transformer_encoder,
         (m, p, s, x) -> first(transformer_encoder_loss(
             m, p, s, x; mask_rate = Float64(opts["mask-rate"]), rng = rng)))
    elseif flavour === :decoder
        ("tx_dec", :transformer_decoder, transformer_decoder,
         (m, p, s, x) -> first(transformer_decoder_loss(m, p, s, x)))
    else
        throw(ArgumentError("unknown transformer flavour `$flavour`"))
    end

    model = factory(vocab_size;
                    d_model     = dims.d_model,
                    n_layers    = Int(opts["n-layers"]),
                    n_heads     = dims.n_heads,
                    n_kv_heads  = dims.n_kv_heads,
                    ffn_mult    = Float64(opts["ffn-mult"]),
                    max_seq_len = max_seq_len,
                    dropout     = Float64(opts["dropout"]))
    ps, st = Lux.setup(rng, model)
    ps = _sgd_tx!(model, ps, st, S, loss_fn, Int(opts["epochs"]),
                  Int(opts["batch"]), Float32(opts["lr"]), rng;
                  quiet = opts["quiet"], label = label)

    save_kwargs = (
        kind = kind, vocab_size = vocab_size, d_model = dims.d_model,
        n_layers = Int(opts["n-layers"]), n_heads = dims.n_heads,
        n_kv_heads = dims.n_kv_heads,
        ffn_mult = Float32(opts["ffn-mult"]),
        max_seq_len = max_seq_len, dropout = Float32(opts["dropout"]),
        vocab = vocab, seqlen = seqlen,
        n_lines = length(lines),
        metadata = _train_metadata(opts, lines, String(label)),
    )
    PersistenceGlue.save(opts["out"], model, ps, st; save_kwargs...)
    opts["quiet"] || println(stderr,
        rpad(label, 9), length(vocab), "-tok vocab, d_model=", dims.d_model,
        ", layers=", Int(opts["n-layers"]),
        ", heads=", dims.n_heads, "/", dims.n_kv_heads,
        ", seqlen=", seqlen, " → ", opts["out"])
    return 0
end

"""
    _sgd_tx!(model, ps, st, S, loss_fn, epochs, batch, lr, rng; …)

Generic Zygote pullback loop over `(seqlen, batch)` integer-id mini-
batches. `loss_fn(model, ps, st, batch) -> scalar` so the same loop
serves both encoder MLM and decoder shift-one objectives.
"""
function _sgd_tx!(model, ps, st, S::AbstractMatrix{<:Integer}, loss_fn,
                  epochs::Int, batch::Int, lr::Float32, rng;
                  quiet::Bool = false,
                  label::AbstractString = "transformer")
    n = size(S, 2)
    batch = min(batch, n)
    for epoch in 1:epochs
        perm = randperm(rng, n)
        total = 0.0f0
        n_batches = 0
        for start in 1:batch:n
            stop = min(start + batch - 1, n)
            cols = perm[start:stop]
            Sb = S[:, cols]
            (loss, back) = Zygote.pullback(p -> loss_fn(model, p, st, Sb), ps)
            g = back(one(loss))[1]
            ps = _apply_sgd!(ps, g, lr)
            total += loss
            n_batches += 1
        end
        if !quiet && (epoch % max(1, epochs ÷ 5) == 0 || epoch == epochs)
            println(stderr, rpad(label, 9), "  epoch ", lpad(epoch, 3),
                    "/", epochs, "   loss=",
                    round(total / max(1, n_batches); digits = 4))
        end
    end
    return ps
end

# --- Shared SGD helpers -----------------------------------------------------

"Apply an SGD step over a NamedTuple / Array parameter tree."
function _apply_sgd!(ps, grads, lr)
    grads === nothing && return ps
    if ps isa AbstractArray
        return ps .- lr .* grads
    elseif ps isa NamedTuple
        ks = keys(ps)
        return NamedTuple{ks}(map(k -> _apply_sgd!(getfield(ps, k),
                                                   hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                   lr), ks))
    else
        return ps
    end
end

"BCE-reconstruction SGD loop (DeepKATE)."
function _sgd_recon!(model, ps, st, X::AbstractMatrix, epochs::Int, batch::Int,
                     lr::Float32, rng; quiet::Bool = false,
                     label::AbstractString = "model")
    n = size(X, 2)
    batch = min(batch, n)
    # Delegate the loss body to `DeepKATE.deep_kate_loss` (4-arg form,
    # reconstruction BCE only). Any Lux `Chain` that's `input-in / input-
    # out` shape-preserving works, including DeepKATE itself. State `st`
    # is passed in training mode so Dropout + KATE competition both fire
    # inside the autodiff step — Lux emits a warning otherwise.
    for epoch in 1:epochs
        perm = randperm(rng, n)
        total = 0.0f0
        n_batches = 0
        for start in 1:batch:n
            stop = min(start + batch - 1, n)
            cols = perm[start:stop]
            Xb = X[:, cols]
            (loss, back) = Zygote.pullback(ps) do p
                first(DeepKATE.deep_kate_loss(model, p, st, Xb))
            end
            g = back(one(loss))[1]
            ps = _apply_sgd!(ps, g, lr)
            total += loss
            n_batches += 1
        end
        if !quiet && (epoch % max(1, epochs ÷ 5) == 0 || epoch == epochs)
            println(stderr, rpad(label, 9), "  epoch ", lpad(epoch, 3),
                    "/", epochs, "   loss=",
                    round(total / max(1, n_batches); digits = 4))
        end
    end
    return ps
end

"VQ-VAE-specific SGD loop (uses the reconstruction + codebook + commitment loss)."
function _sgd_vqvae!(model, ps, st, X::AbstractMatrix, epochs::Int, batch::Int,
                     lr::Float32, rng; quiet::Bool = false)
    n = size(X, 2)
    batch = min(batch, n)
    for epoch in 1:epochs
        perm = randperm(rng, n)
        total = 0.0f0
        n_batches = 0
        for start in 1:batch:n
            stop = min(start + batch - 1, n)
            cols = perm[start:stop]
            Xb = X[:, cols]
            (loss, back) = Zygote.pullback(p ->
                first(VQVAE.vq_vae_loss(model, p, st, Xb)), ps)
            g = back(one(loss))[1]
            ps = _apply_sgd!(ps, g, lr)
            total += loss
            n_batches += 1
        end
        if !quiet && (epoch % max(1, epochs ÷ 5) == 0 || epoch == epochs)
            println(stderr, "vq_vae    epoch ", lpad(epoch, 3),
                    "/", epochs, "   loss=",
                    round(total / max(1, n_batches); digits = 4))
        end
    end
    return ps
end

"SeqLSTM next-event-NLL SGD loop over `(seqlen, batch)` integer id matrix."
function _sgd_seqlstm!(model, ps, st, S::AbstractMatrix{<:Integer}, epochs::Int,
                       batch::Int, lr::Float32, rng; quiet::Bool = false)
    n = size(S, 2)
    batch = min(batch, n)
    for epoch in 1:epochs
        perm = randperm(rng, n)
        total = 0.0f0
        n_batches = 0
        for start in 1:batch:n
            stop = min(start + batch - 1, n)
            cols = perm[start:stop]
            Sb = S[:, cols]
            (loss, back) = Zygote.pullback(p ->
                first(seq_lstm_loss(model, p, st, Sb)), ps)
            g = back(one(loss))[1]
            ps = _apply_sgd!(ps, g, lr)
            total += loss
            n_batches += 1
        end
        if !quiet && (epoch % max(1, epochs ÷ 5) == 0 || epoch == epochs)
            println(stderr, "seq_lstm  epoch ", lpad(epoch, 3),
                    "/", epochs, "   loss=",
                    round(total / max(1, n_batches); digits = 4))
        end
    end
    return ps
end

# We import VQVAE here because `vq_vae_loss` is referenced via the
# module name to keep the `using` list tidy.
using ..VQVAE: VQVAE

# `randperm` is used above but not re-exported by default when the
# imports skip `Random`.
using Random: randperm

function _print_train_help()
    println("""
    usage: logcluster train --kind KIND --data FILE --out FILE
                            [--auto] [--budget N] [--epochs N] [--batch N]
                            [--lr F] [--seed N]
                            [--seqlen N] [--max-vocab N] [--min-count N]
                            [--reuse] [--registry DIR]
                            [--quiet]

    Fit and persist a model.

    KIND:
      drain                streaming log-template parser (no featurisation needed)
      deep_kate            Lux autoencoder on a log-normalised BoW matrix
      vq_vae               Lux codebook AE on a binary BoW matrix
      seq_lstm             Lux next-event LSTM on padded token-id sequences
      transformer_encoder  Pre-Norm transformer (bidirectional) trained MLM-style
      transformer_decoder  Pre-Norm causal transformer trained shift-one LM
                           Both transformer kinds use RoPE + RMSNorm + GQA + SwiGLU.

    All AE/LSTM/transformer kinds build a typed-slot-masked vocabulary
    on the training corpus and persist it in the JLD2 bundle, so a
    later `classify` pass can re-featurise new lines identically.

    --auto        run AutoTune.fit_hyperparams on the featurised corpus.
    --budget N    >0 triggers random search around the heuristic seed.
    --epochs N    training epochs (default 20).
    --batch N     mini-batch size (default 64).
    --lr F        SGD learning rate (default 0.05).
    --seed N      RNG seed (default 0).
    --seqlen N    sequence length for seq_lstm (default 16).
    --max-vocab N cap the vocabulary (default 5000).
    --min-count N minimum token frequency to keep (default 1).
    --reuse       skip training if --registry holds a bundle whose
                  `corpus_fingerprint` matches this corpus's vocab;
                  the bundle is then copied to --out. Ignored for
                  --kind drain (drain bundles don't carry a vocab).
    --registry D  directory of saved bundles to scan for --reuse and
                  `select-model`. Default: \$XDG_CACHE_HOME/logclustering/models
                  (or ~/.cache/logclustering/models).
    --quiet       suppress per-epoch loss lines.

    Transformer-only flags (ignored for other kinds):
      --d-model N     hidden width (default 128).
      --n-layers N    number of transformer blocks (default 4).
      --n-heads N     query heads (default 8).
      --n-kv-heads N  KV heads for GQA (default n-heads ÷ 4, min 1;
                      0 = use the default).
      --ffn-mult F    SwiGLU width multiplier (default 4).
      --dropout F     post-attention / post-FFN dropout (default 0).
      --mask-rate F   MLM mask rate, encoder only (default 0.15).
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
        ("json",   false, :bool),            # alias for --format json
        ("framed", false, :bool),
        ("oov",           "unk", :string),   # unk | nearest | distribute
        ("oov-min-sim",   0.3,   :float),
        ("oov-top-k",     3,     :int),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_classify_help(); return 0)
    opts["json"] && (opts["format"] = "json")
    isempty(opts["model"]) && throw(ArgumentError("--model is required"))
    _validate_oov_policy(opts)
    bundle = Persistence.load(opts["model"])
    artifact = Persistence.rehydrate(bundle)

    lines = read_lines(opts["data"])
    bodies = opts["framed"] ? [String(parse_frame(l).message) for l in lines] : lines

    if bundle.kind === :drain
        ids = Int[]
        templates = String[]
        for l in bodies
            cid, tpl = process!(artifact, l)
            push!(ids, cid); push!(templates, tpl)
        end
        _emit_classify(opts["out"], opts["format"], bodies, ids, templates)
        return 0
    elseif bundle.kind === :deep_kate
        return _classify_deep_kate(artifact, bundle, bodies, opts)
    elseif bundle.kind === :vq_vae
        return _classify_vq_vae(artifact, bundle, bodies, opts)
    elseif bundle.kind === :seq_lstm
        return _classify_seq_lstm(artifact, bundle, bodies, opts)
    elseif bundle.kind === :transformer_encoder
        return _classify_transformer_encoder(artifact, bundle, bodies, opts)
    elseif bundle.kind === :transformer_decoder
        return _classify_transformer_decoder(artifact, bundle, bodies, opts)
    else
        throw(ArgumentError(
            "model kind $(bundle.kind) not yet handled by classify"))
    end
end

"""
    _validate_oov_policy(opts)

Throw a clear `ArgumentError` when `--oov` is misspelt. `:distribute`
is BoW-only; `sequence_matrix`-driven kinds (e.g. seq_lstm) reject
it downstream but we catch the obvious typos here.
"""
function _validate_oov_policy(opts)
    v = String(opts["oov"])
    v in ("unk", "nearest", "distribute") ||
        throw(ArgumentError("--oov must be one of unk | nearest | distribute; got `$v`"))
end

@inline function _oov_kwargs(opts)
    (oov_policy = Symbol(opts["oov"]),
     oov_min_sim = Float64(opts["oov-min-sim"]),
     oov_top_k   = Int(opts["oov-top-k"]))
end

function _require_vocab(artifact, kind)
    artifact.vocab === nothing && throw(ArgumentError(
        "`$kind` bundle has no vocabulary; save it with " *
        "`PersistenceGlue.save(path, model, ps, st; kind=:$kind, …, vocab=v)`"))
    return artifact.vocab
end

# DeepKATE clusters via the thesis's sparsity-from-argmax signal: we
# run the encoder in testmode, take the top-`k` active latent slots
# per sample (K = 1 here — the most active neuron is the cluster id),
# and emit `(cluster_id, "cluster-<id>")`. Drain's richer "template
# string per cluster" representation would need a separate decoder
# pass + template voting, which the CLI currently stops short of.
function _classify_deep_kate(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :deep_kate)
    X = bow(bodies, vocab; normalise = :l1, _oov_kwargs(opts)...)
    latent_idx = length(art.model.layers) - 5   # position of the sine bottleneck in deep_kate
    # Actually reach through to the Lux internals: thesis DeepKATE
    # has `latent_layer == 5`, the Dense(5 => latent, sin) output.
    Z = _forward_through(art.model, art.ps, Lux.testmode(art.st), X, 5)
    assignments, _ = sparsity_clusters(Z, 1)
    labels = ["cluster-$(i)" for i in assignments]
    _emit_classify(opts["out"], opts["format"], bodies, assignments, labels)
    return 0
end

function _classify_vq_vae(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :vq_vae)
    X = bow(bodies, vocab; normalise = :binary, _oov_kwargs(opts)...)
    codes = assign_codes(art.model, art.ps, art.st, X)
    labels = ["code-$(c)" for c in codes]
    _emit_classify(opts["out"], opts["format"], bodies, codes, labels)
    return 0
end

function _classify_seq_lstm(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :seq_lstm)
    seqlen = art.seqlen === nothing ? 16 : Int(art.seqlen)
    oov_policy = Symbol(opts["oov"])
    oov_policy === :distribute &&
        throw(ArgumentError(":distribute is BoW-only; seq_lstm needs :unk or :nearest"))
    S = sequence_matrix(bodies, vocab;
                        seqlen = seqlen,
                        oov_policy = oov_policy,
                        oov_min_sim = Float64(opts["oov-min-sim"]))
    # seq_lstm output is (vocab_size, batch) logits — argmax = predicted
    # next token's id. We emit that as the "cluster id"; the template
    # is the predicted token string.
    logits, _ = art.model(S, art.ps, Lux.testmode(art.st))
    ids = Int[argmax(@view logits[:, j]) for j in axes(logits, 2)]
    labels = [get(vocab.tokens, id, "<UNK>") for id in ids]
    _emit_classify(opts["out"], opts["format"], bodies, ids, labels)
    return 0
end

# Encoder: mean-pool the post-norm hidden states, kmeans on the
# `(d_model, batch)` embedding. Cluster count defaults to a √N rule
# bounded to the same band the rest of the pipeline uses.
function _classify_transformer_encoder(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :transformer_encoder)
    seqlen = art.seqlen === nothing ? 64 : Int(art.seqlen)
    oov_policy = Symbol(opts["oov"])
    oov_policy === :distribute &&
        throw(ArgumentError(":distribute is BoW-only; transformer_encoder needs :unk or :nearest"))
    S = sequence_matrix(bodies, vocab;
                        seqlen = seqlen,
                        oov_policy = oov_policy,
                        oov_min_sim = Float64(opts["oov-min-sim"]))
    Z = embed_sequences(art.model, art.ps, art.st, S)            # (d_model, batch)
    n_lines = size(Z, 2)
    k = min(n_lines, max(2, ceil(Int, sqrt(n_lines))))
    Zn = Pipeline.l2_normalise(Z)
    res = Pipeline.kmeans_cluster(Zn, k)
    assignments = collect(res.assignments)
    labels = ["cluster-$(i)" for i in assignments]
    _emit_classify(opts["out"], opts["format"], bodies, assignments, labels)
    return 0
end

# Decoder: greedy next-token prediction (mirrors seq_lstm's path so
# the output schema matches downstream consumers).
function _classify_transformer_decoder(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :transformer_decoder)
    seqlen = art.seqlen === nothing ? 64 : Int(art.seqlen)
    oov_policy = Symbol(opts["oov"])
    oov_policy === :distribute &&
        throw(ArgumentError(":distribute is BoW-only; transformer_decoder needs :unk or :nearest"))
    S = sequence_matrix(bodies, vocab;
                        seqlen = seqlen,
                        oov_policy = oov_policy,
                        oov_min_sim = Float64(opts["oov-min-sim"]))
    ids = Transformer.predict_next(art.model, art.ps, art.st, S)
    labels = [get(vocab.tokens, id, "<UNK>") for id in ids]
    _emit_classify(opts["out"], opts["format"], bodies, ids, labels)
    return 0
end

"Run `model`'s first `lat` layers in sequence, returning the activation."
function _forward_through(model, ps, st, X, lat::Int)
    out = X
    for i in 1:lat
        sym = Symbol(:layer_, i)
        layer = getfield(model.layers, sym)
        p = getfield(ps, sym)
        s = getfield(st, sym)
        out, _ = layer(out, p, s)
    end
    return out
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
                               [--oov unk|nearest|distribute]
                               [--oov-min-sim F] [--oov-top-k N]

    Load a saved model and emit one record per input line:

      drain / deep_kate / vq_vae / seq_lstm        — existing paths
      transformer_encoder                          — kmeans on mean-pooled
                                                    post-norm hidden states
      transformer_decoder                          — greedy next-token argmax
    """)
end

# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

function cmd_score(args::Vector{String})::Int
    specs = [
        ("detector", "",   :path),
        ("model",    "",   :path),         # transformer_decoder bundle for NLL scoring
        ("data",     "-",  :path),
        ("out",      "-",  :path),
        ("format",   "tsv", :string),
        ("json",     false, :bool),         # alias for --format json
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_score_help(); return 0)
    opts["json"] && (opts["format"] = "json")

    have_det   = !isempty(opts["detector"])
    have_model = !isempty(opts["model"])
    (have_det || have_model) ||
        throw(ArgumentError("--detector or --model is required"))

    lines = read_lines(opts["data"])

    # Per-line scores from each requested source. When both are
    # present, sum them — `score(d, m, x) = value_novelty(d) + nll(m)`.
    scores = zeros(Float64, length(lines))

    if have_det
        det = Persistence.load_and_rehydrate(opts["detector"])
        det isa ValueNoveltyDetector ||
            throw(ArgumentError("--detector must be a ValueNoveltyDetector bundle"))
        _, values = mask_lines_with_values(lines)
        for (i, vs) in enumerate(values)
            scores[i] += Float64(value_novelty(det, vs))
        end
    end

    if have_model
        bundle = Persistence.load(opts["model"])
        bundle.kind === :transformer_decoder ||
            throw(ArgumentError(
                "--model must be a :transformer_decoder bundle for score; got $(bundle.kind)"))
        art = Persistence.rehydrate(bundle)
        nlls = _per_line_decoder_nll(art, lines)
        for i in eachindex(lines)
            scores[i] += Float64(nlls[i])
        end
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

"""
    _per_line_decoder_nll(art, lines) -> Vector{Float32}

Compute per-line negative log-likelihood from a `:transformer_decoder`
bundle. Higher = the model finds the line less typical (the standard
perplexity-style anomaly signal). Each line is scored independently so
batch ordering doesn't bleed across lines.
"""
function _per_line_decoder_nll(art, lines::Vector{String})::Vector{Float32}
    vocab = _require_vocab(art, :transformer_decoder)
    seqlen = art.seqlen === nothing ? 64 : Int(art.seqlen)
    nlls = Vector{Float32}(undef, length(lines))
    for (i, l) in enumerate(lines)
        S = sequence_matrix([l], vocab; seqlen = seqlen)
        loss, _ = transformer_decoder_loss(art.model, art.ps,
                                            Lux.testmode(art.st), S)
        nlls[i] = Float32(loss)
    end
    return nlls
end

"""
    _decoder_nll_batch(art, lines) -> Vector{Float64}

Batched per-line decoder NLL: one `sequence_matrix(lines)` + one
`transformer_decoder_nll` forward for the whole batch. Because the
decoder attends only within a column, this is byte-identical to
`_per_line_decoder_nll` called once per line — it just amortizes the
forward. Used by the streaming micro-batch path.
"""
function _decoder_nll_batch(art, lines::Vector{String})::Vector{Float64}
    isempty(lines) && return Float64[]
    vocab = _require_vocab(art, :transformer_decoder)
    seqlen = art.seqlen === nothing ? 64 : Int(art.seqlen)
    S = sequence_matrix(lines, vocab; seqlen = seqlen)
    return transformer_decoder_nll(art.model, art.ps,
                                   Lux.testmode(art.st), S)
end

function _print_score_help()
    println("""
    usage: logcluster score [--detector PATH] [--model PATH]
                            [--data FILE] [--out FILE]
                            [--format tsv|json]

    Per-line anomaly score. At least one signal source is required:

      --detector PATH  ValueNoveltyDetector bundle — slot-novelty term.
      --model PATH     :transformer_decoder bundle — per-line NLL
                       (perplexity proxy). Higher = less typical.

    When both are supplied the per-line scores are summed.
    """)
end

# ---------------------------------------------------------------------------
# benchmark / download-loghub — thin wrappers that delegate to the existing
# scripts under `benchmarks/loghub2/`.
# ---------------------------------------------------------------------------

function cmd_select(args::Vector{String})::Int
    specs = [
        ("kind",      "deep_kate", :string),
        ("data",      "-",         :path),
        ("registry",  _default_registry_path(), :path),
        ("max-vocab", 5000,        :int),
        ("min-count", 1,           :int),
        ("format",    "path",      :string),   # "path" | "info"
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_select_help(); return 0)
    kind = Symbol(opts["kind"])
    lines = read_lines(opts["data"])
    vocab = build_vocab(lines; mask = true,
                        min_count = Int(opts["min-count"]),
                        max_vocab = Int(opts["max-vocab"]))
    fp = Persistence.corpus_fingerprint(vocab.tokens; n_lines = length(lines))
    candidate = Persistence.find_compatible_bundle(
        String(opts["registry"]), kind, fp)
    if candidate === nothing
        println(stderr,
                "no ", kind, " bundle matches fingerprint ", fp[1:12], "…",
                " in ", opts["registry"])
        return 1
    end
    if opts["format"] == "info"
        println(candidate.path)
        println("  kind=", candidate.kind,
                "  schema=", Int(candidate.schema_version))
        println("  n_lines=", get(candidate.metadata, "n_lines", "?"),
                "  source=", get(candidate.metadata, "source", "?"))
    else
        println(candidate.path)
    end
    return 0
end

function _print_select_help()
    println("""
    usage: logcluster select-model [--kind KIND] [--data FILE]
                                   [--registry DIR] [--format path|info]

    Print the path of a saved bundle in --registry whose
    `corpus_fingerprint` matches the vocab of --data. Exits 1 when no
    match is found. Use `--format info` to include kind + metadata.
    """)
end

function cmd_benchmark(args::Vector{String})::Int
    # Lazy-include to avoid hard-wiring the path at CLI module load time.
    here = @__DIR__
    run_jl = joinpath(here, "..", "benchmarks", "loghub2", "run.jl")
    isfile(run_jl) ||
        throw(ArgumentError("benchmark script not found at $run_jl"))
    # `run.jl` reads `ARGS` directly; hand it our tail and call its main.
    # Fresh anonymous module keeps the script's globals out of CLI's
    # namespace, but `Module(:X)` doesn't expose `include` — use
    # `Base.include` explicitly.
    mod = Module(:LcBenchmarkShim)
    Core.eval(mod, :(ARGS = $(copy(args))))
    Base.include(mod, run_jl)
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
    Base.include(mod, dl_jl)
    Core.eval(mod, :(main(ARGS)))
    return 0
end

# ---------------------------------------------------------------------------
# Subcommand table.
# ---------------------------------------------------------------------------

function cmd_rca(args::Vector{String})::Int
    specs = [
        ("model",       "",       :path),
        ("detector",    "",       :path),
        ("data",        "-",      :path),
        ("out",         "-",      :path),
        ("format",      "md",     :string),       # md | json | tsv
        ("topk",        10,       :int),
        ("percentile",  0.10,     :float),
        ("min-sup",     3,        :int),
        ("max-gap",     20,       :int),
        ("max-dur",     50,       :int),
        ("k-clusters",  0,        :int),          # 0 = auto
        ("embedder",    "model",  :string),       # model | sparsity
        ("sparsity-k",  5,        :int),
        ("max-vocab",   5000,     :int),
        ("min-count",   1,        :int),
        ("oov",           "unk", :string),   # unk | nearest | distribute
        ("oov-min-sim",   0.3,   :float),
        ("oov-top-k",     3,     :int),
        ("json",          false, :bool),       # alias for --format json
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_rca_help(); return 0)
    opts["json"] && (opts["format"] = "json")
    _validate_oov_policy(opts)
    lines = read_lines(opts["data"])

    detector = nothing
    if !isempty(opts["detector"])
        d = Persistence.load_and_rehydrate(opts["detector"])
        d isa ValueNoveltyDetector ||
            throw(ArgumentError("--detector must be a ValueNoveltyDetector bundle"))
        detector = d
    end

    embedder = Symbol(opts["embedder"])
    report = if embedder === :sparsity
        # No model, no training. Build the vocab from the corpus itself.
        vocab = build_vocab(lines; mask = true,
                            min_count = Int(opts["min-count"]),
                            max_vocab = Int(opts["max-vocab"]))
        root_cause_sparsity(vocab, lines;
            sparsity_k = Int(opts["sparsity-k"]),
            detector = detector,
            top_percentile = Float64(opts["percentile"]),
            min_sup = Int(opts["min-sup"]),
            max_gap = Int(opts["max-gap"]),
            max_time_duration = Int(opts["max-dur"]))
    elseif embedder === :model
        isempty(opts["model"]) &&
            throw(ArgumentError("--model is required for --embedder model"))
        art = Persistence.load_and_rehydrate(opts["model"])
        hasproperty(art, :model) && hasproperty(art, :vocab) ||
            throw(ArgumentError("--model must be a DeepKATE / VQ-VAE bundle"))
        kk = Int(opts["k-clusters"])
        root_cause(art.model, art.ps, art.st, art.vocab, lines;
            detector = detector,
            k_clusters = kk > 0 ? kk : nothing,
            top_percentile = Float64(opts["percentile"]),
            min_sup = Int(opts["min-sup"]),
            max_gap = Int(opts["max-gap"]),
            max_time_duration = Int(opts["max-dur"]),
            _oov_kwargs(opts)...)
    else
        throw(ArgumentError("unknown --embedder `$(opts["embedder"])`; use `model` or `sparsity`"))
    end

    body = if opts["format"] == "md"
        render_markdown(report; topk = Int(opts["topk"]), lines = lines)
    elseif opts["format"] == "json"
        JSON3.write(Dict(
            "metadata" => report.metadata,
            "cluster_ids" => report.cluster_ids,
            "per_line_score" => report.per_line_score,
            "ranked" => [Dict("pattern" => r.pattern, "support" => r.support,
                              "density" => r.density, "score" => r.score)
                         for r in report.ranked[1:min(end, Int(opts["topk"]))]],
        ))
    elseif opts["format"] == "tsv"
        io = IOBuffer()
        println(io, "rank\tpattern\tsupport\tdensity\tscore")
        for (i, r) in enumerate(report.ranked)
            i > Int(opts["topk"]) && break
            println(io, i, '\t', join(r.pattern, ','), '\t',
                    r.support, '\t', r.density, '\t', r.score)
        end
        String(take!(io))
    else
        throw(ArgumentError("unknown --format `$(opts["format"])`"))
    end
    write_out(opts["out"], body)
    return 0
end

function _print_rca_help()
    println("""
    usage: logcluster rca [--embedder model|sparsity]
                          [--model PATH] [--detector PATH]
                          [--data FILE] [--out FILE]
                          [--format md|json|tsv]
                          [--topk N] [--percentile F]
                          [--min-sup N] [--max-gap N] [--max-dur N]
                          [--k-clusters K] [--sparsity-k K]
                          [--max-vocab N] [--min-count N]

    Root-cause-analysis report: cluster → anomaly-score → episode
    mining, seeded with the cluster ids of the top-percentile most
    anomalous lines.

    Embedders:
      model     (default) run a saved DeepKATE / VQ-VAE bundle's
                encoder, k-means on the latent. Requires --model.
      sparsity  no training: cluster via top-k active BoW tokens
                per line (`Cluster.Sparsity.sparsity_clusters` on
                L2-normalised raw BoW). Tune with --sparsity-k
                (default 5). Empirically matches or beats the
                model path on dense-vocabulary corpora.

    --detector is optional; it fuses a saved ValueNoveltyDetector's
    signal with the per-line anomaly score. Without a detector the
    `model` path uses reconstruction error; the `sparsity` path
    uses -log(p(cluster_id)) as a frequency proxy.

    The Markdown format includes a table of the top-N root-cause
    episodes (pattern, support, density, score) and representative
    lines from each.
    """)
end

# ---------------------------------------------------------------------------
# stream — long-running line-by-line inference + rule evaluation.
# ---------------------------------------------------------------------------

"""
    StreamInferer

Bundle holding the loaded model + optional detector + framing mode
for the streaming worker. One per process; reused for every line so
allocations stay bounded.
"""
struct StreamInferer
    bundle_kind::Union{Symbol, Nothing}
    artifact::Any
    detector::Any              # ::Union{ValueNoveltyDetector, Nothing}
    framed::Symbol             # :raw | :auto
    nll_memo::Any              # ::Union{Nothing, Dedup.LRUMemo} — near-dup reuse
end

function cmd_stream(args::Vector{String})::Int
    specs = [
        ("model",              "",     :path),
        ("detector",           "",     :path),
        ("rules",              "",     :path),
        ("data",               "",     :path),       # "" or "-" = stdin
        ("tail",               false,  :bool),
        ("framed",             "raw",  :string),     # raw | auto
        ("emit-all",           false,  :bool),
        ("warmup-lines",       500,    :int),
        ("warmup-seconds",     30.0,   :float),
        ("status-interval",    10.0,   :float),
        ("max-line-bytes",     65_536, :int),
        ("webhook",            "",     :string),
        ("webhook-secret-env", "",     :string),
        ("webhook-format",     "raw",  :string),     # raw | slack | alertmanager
        ("quiet",              false,  :bool),
        ("exit-on-trigger",    false,  :bool),
        ("shutdown-timeout",   5.0,    :float),
        ("log-format",         "json", :string),     # json | text
        ("log-level",          "info", :string),     # debug | info | warn | error
        ("max-events",         0,      :int),        # 0 = no limit (test hook)
        ("channel-capacity",   4096,   :int),
        ("memory",             "",     :path),       # SQLite filename for triggers
        ("persist-lines",      "none", :string),     # none | sampled | all
        ("persist-lines-rate", 100,    :int),        # 1-in-N sampling rate
        ("memory-warm",        "",     :string),     # redis://... | inproc | ""
        ("memory-warm-namespace", "",  :string),     # override the auto namespace
        ("memory-warm-flush-on-boot", false, :bool),
        ("use-patterns",       "",     :path),       # SQLite path; auto-promote pinned patterns
        ("dedup",              "off",  :string),      # off | masked
        ("dedup-window",       4096,   :int),
        ("batch-lines",        1,      :int),         # micro-batch transformer forward
        ("batch-ms",           0.0,    :float),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_stream_help(); return 0)
    _apply_config!(opts, "stream")
    Symbol(opts["dedup"]) in (:off, :masked) ||
        throw(ArgumentError("--dedup must be off | masked"))

    # Configure logger before anything else so all subsequent diagnostics
    # land in the right place.
    StructuredLog.set_format!(Symbol(opts["log-format"]))
    StructuredLog.set_level!(Symbol(opts["log-level"]))

    framed_mode = Symbol(opts["framed"])
    framed_mode in (:raw, :auto) ||
        throw(ArgumentError("--framed must be raw | auto, got `$(opts["framed"])`"))

    # Pre-validate input files so a missing bundle / source / rules
    # file exits with the documented I/O code (3) up front, rather
    # than surfacing as an opaque ArgumentError → 2 deep in the loop.
    for (flag, label) in (("model", "model bundle"),
                          ("detector", "detector bundle"),
                          ("rules", "rules file"),
                          ("use-patterns", "patterns db"))
        p = String(opts[flag])
        if !isempty(p) && !isfile(p)
            StructuredLog.error_event("input not found";
                                      flag = "--$flag", path = p)
            return 3
        end
    end
    src = String(opts["data"])
    if !isempty(src) && src != "-" && !isfile(src)
        StructuredLog.error_event("input not found";
                                  flag = "--data", path = src)
        return 3
    end

    # Load rules; override with --webhook if supplied (synthesises a
    # `webhook:cli` sink + an all-severity route).
    rs = if isempty(opts["rules"])
        Rules.default_rules(warmup_lines = Int(opts["warmup-lines"]),
                            warmup_seconds = Float64(opts["warmup-seconds"]))
    else
        Rules.load_rules(String(opts["rules"]);
                         warmup_lines = Int(opts["warmup-lines"]),
                         warmup_seconds = Float64(opts["warmup-seconds"]))
    end

    # Auto-promotion of pinned patterns into rules + drain sidecar.
    drain_pinned = Memory.PatternCatalog.Pattern[]
    if !isempty(opts["use-patterns"])
        pdb = Memory.SQLite.open_db(String(opts["use-patterns"]); create = false)
        Memory.SQLite.migrate!(pdb)
        pinned = Memory.PatternCatalog.list(pdb; enabled_only = true)
        for r in Memory.PatternCatalog.to_rules(pinned)
            push!(rs.rules, r)
            rs.state[r.id] = Rules._initial_state(r)
        end
        drain_pinned = Memory.PatternCatalog.drain_patterns(pinned)
        StructuredLog.info("patterns loaded";
                           total = length(pinned),
                           synthesised_rules = length(rs.rules),
                           drain_patterns = length(drain_pinned))
        try; close(pdb); catch; end
    end

    inferer = _build_inferer(opts, framed_mode)
    _warn_inert_rules(rs, inferer)

    # Optional short-term warm state (Redis): hydrates novel-cluster
    # baselines so a restart doesn't re-fire every template the
    # operator has already seen.
    if !isempty(opts["memory-warm"])
        ns = isempty(opts["memory-warm-namespace"]) ?
             _rules_fingerprint(opts) :
             String(opts["memory-warm-namespace"])
        try
            warm = Memory.WarmStore.open_warm(String(opts["memory-warm"]);
                rules_fingerprint = ns)
            if opts["memory-warm-flush-on-boot"]
                for r in rs.rules
                    r isa Rules.NovelClusterRule || r isa Rules.NovelTokenRule || continue
                    Memory.WarmStore.flush_rule!(warm, r.id)
                end
                StructuredLog.info("warm store flushed on boot"; namespace = ns)
            end
            Rules.attach_warm_store!(rs, warm)
            StructuredLog.info("warm store attached";
                               url = String(opts["memory-warm"]),
                               namespace = ns)
        catch e
            StructuredLog.error_event("warm store open failed";
                                      url = String(opts["memory-warm"]),
                                      error = sprint(showerror, e))
            return 3
        end
    end

    # Optional long-term memory: SQLite writer + session row. The
    # async writer absorbs back-pressure so the inference loop never
    # waits on disk I/O.
    persist_lines_mode = Symbol(opts["persist-lines"])
    persist_lines_mode in (:none, :sampled, :all) ||
        throw(ArgumentError("--persist-lines must be none | sampled | all"))
    persist_lines_rate = max(1, Int(opts["persist-lines-rate"]))
    mem_db = nothing
    mem_ch = nothing
    mem_task = nothing
    mem_stop = nothing
    mem_stats = nothing
    mem_session_id = 0
    if !isempty(opts["memory"])
        mem_db = Memory.SQLite.open_db(String(opts["memory"]))
        Memory.SQLite.migrate!(mem_db)
        mem_session_id = Memory.SQLite.insert_session!(mem_db;
            host          = gethostname(),
            model_path    = String(opts["model"]),
            detector_path = String(opts["detector"]),
            rules_path    = isempty(opts["rules"]) ? "<defaults>" : String(opts["rules"]),
            rules_sha256  = "",
            meta = Dict{String, Any}(
                "framed"          => String(opts["framed"]),
                "warmup_lines"    => Int(opts["warmup-lines"]),
                "warmup_seconds"  => Float64(opts["warmup-seconds"]),
                "persist_lines"   => String(opts["persist-lines"]),
            ))
        mem_ch, mem_task, mem_stop, mem_stats = Memory.SQLite.spawn_writer(mem_db;
            batch = 100, flush_ms = 250)
    end

    # Optional webhook sink injected at runtime from --webhook (the
    # rules JSON's existing webhook sinks still apply alongside).
    # Synthesise an all-severity route to the `webhook:cli` sink so
    # every fired rule — not just crit ones already routed in the
    # rules file — reaches the CLI webhook.
    webhook_sink = _maybe_build_webhook_sink(opts)
    webhook_ch::Union{Nothing, Channel{Dict{String, Any}}} = nothing
    webhook_task::Union{Nothing, Task} = nothing
    webhook_stop::Union{Nothing, Ref{Bool}} = nothing
    if webhook_sink !== nothing
        push!(rs.routes, Rules.Route(Set{Symbol}(), Set{String}(),
                                     ["webhook:cli"]))
        webhook_ch, webhook_task, webhook_stop =
            SinksWebhook.start_webhook_task(webhook_sink; capacity = 1024)
    end

    # Producer task.
    ch = Channel{Stream.LineEvent}(Int(opts["channel-capacity"]))
    stop_signal = Stream.StopSignal()
    tail_stats = Stream.TailStats()
    producer = _spawn_producer(opts, ch, stop_signal, tail_stats)

    started_at = time()
    last_status = started_at
    triggers_total = 0
    by_rule = Dict{String, Int}()
    lines_processed = 0
    mem_writes_dropped = 0
    producer_failed = false
    drain_last_fired = Dict{String, Float64}()   # sidecar cooldown clock
    max_events = Int(opts["max-events"])
    status_interval = Float64(opts["status-interval"])
    exit_on_trigger = opts["exit-on-trigger"]
    batch_lines = max(1, Int(opts["batch-lines"]))
    batch_ms    = max(0.0, Float64(opts["batch-ms"]))

    StructuredLog.info("stream started";
                       model = String(opts["model"]),
                       detector = String(opts["detector"]),
                       rules = isempty(opts["rules"]) ? "<defaults>" : String(opts["rules"]),
                       source = isempty(opts["data"]) || opts["data"] == "-" ? "stdin" : opts["data"],
                       tail = opts["tail"])

    # SIGINT (which the systemd unit / Docker STOPSIGNAL send on stop)
    # raises InterruptException into this task instead of hard-killing
    # the process, so the finally block can drain channels + finalize
    # the session. There is no getter to restore the prior setting;
    # leaving it disabled is the intended long-running-daemon
    # behaviour and is harmless for in-process (test) callers.
    Base.exit_on_sigint(false)
    shutdown_reason = "eof"

    try
        done = false
        while !done
            # Block for the first event; a closed+drained channel ends
            # the loop. InterruptException (SIGINT) propagates to the
            # outer catch for graceful shutdown.
            local ev0
            try
                ev0 = take!(ch)
            catch e
                e isa InvalidStateException && break   # channel closed
                rethrow()
            end
            # Opportunistic micro-batch: drain whatever is *already*
            # queued (a backlog under load) up to batch_lines, so the
            # transformer forward is amortized — but never wait when the
            # channel is empty, so light load adds zero latency. With
            # batch_ms > 0, wait briefly for a fuller batch under bursty
            # load. batch_lines = 1 (default) = today's behavior exactly.
            evs = Stream.LineEvent[ev0]
            entries = Any[_cheap_or_memo!(inferer, ev0)]
            while length(evs) < batch_lines && isready(ch)
                e2 = try; take!(ch); catch; break; end
                push!(evs, e2); push!(entries, _cheap_or_memo!(inferer, e2))
            end
            if batch_ms > 0 && length(evs) < batch_lines
                deadline = time() + batch_ms / 1000
                while length(evs) < batch_lines && time() < deadline
                    if isready(ch)
                        e2 = try; take!(ch); catch; break; end
                        push!(evs, e2); push!(entries, _cheap_or_memo!(inferer, e2))
                    else
                        sleep(0.001)
                    end
                end
            end

            # One batched transformer forward fills every missing NLL.
            _fill_batch_nll!(inferer, entries)

            # Process each buffered line in input order.
            for k in eachindex(evs)
                ev = evs[k]
                ir = entries[k][1]::Dict{String, Any}
                lines_processed += 1
                triggers = Rules.evaluate(rs, ir)
                # Sidecar: drain-cluster pinned patterns aren't part of the
                # rule engine because the engine has no DrainTemplateRule
                # kind. Match them directly and synthesise TriggerEvent
                # rows so the downstream emit/persist paths see them.
                # Honour each pattern's cooldown_s and route by
                # severity/rule_id so a crit drain pattern reaches webhooks.
                now_t = time()
                for p in drain_pinned
                    Memory.PatternCatalog.match_line(p, ir) || continue
                    rid = "pattern:$(p.id):$(p.name)"
                    if (now_t - get(drain_last_fired, rid, -Inf)) < p.cooldown_s
                        continue
                    end
                    drain_last_fired[rid] = now_t
                    push!(triggers, Rules.TriggerEvent(
                        rid,
                        :drain_pattern,
                        p.severity,
                        Int(get(ir, "line_id", ev.line_id)),
                        String(get(ir, "line", ev.line)),
                        Dict{String, Any}(
                            "pattern_id" => p.id,
                            "match_kind" => "drain",
                            "cluster_id" => get(get(ir, "drain", Dict{String,Any}()),
                                                 "cluster_id", nothing),
                        ),
                        Rules.route_sinks_for(rs; severity = p.severity, rule_id = rid),
                        Dates.now(Dates.UTC),
                    ))
                end
                if !isempty(triggers) || opts["emit-all"]
                    _emit_line_records(ir, ev, triggers, opts)
                    if webhook_ch !== nothing
                        for t in triggers
                            if "webhook:cli" in t.sinks ||
                               any(startswith(s, "webhook:") for s in t.sinks)
                                SinksWebhook.enqueue!(webhook_ch,
                                    _trigger_to_dict(t, ir), webhook_sink)
                            end
                        end
                    end
                end
                if mem_ch !== nothing
                    for t in triggers
                        _enqueue_trigger_write(mem_ch, mem_session_id, t, ir) ||
                            (mem_writes_dropped += 1)
                        pid = _pattern_id_from_rule(t.rule_id)
                        if pid !== nothing
                            _enqueue_pattern_match_write(mem_ch, mem_session_id,
                                                         pid, t) ||
                                (mem_writes_dropped += 1)
                        end
                    end
                    if persist_lines_mode === :all ||
                       (persist_lines_mode === :sampled &&
                        (lines_processed - 1) % persist_lines_rate == 0)
                        _enqueue_line_write(mem_ch, mem_session_id, ev, ir) ||
                            (mem_writes_dropped += 1)
                    end
                end
                for t in triggers
                    triggers_total += 1
                    by_rule[t.rule_id] = get(by_rule, t.rule_id, 0) + 1
                end
                if status_interval > 0 && !opts["quiet"] &&
                   (time() - last_status) >= status_interval
                    _emit_status(started_at, lines_processed, triggers_total,
                                 by_rule, tail_stats, ch, inferer)
                    Rules.persist_sketches!(rs)
                    last_status = time()
                end
                if max_events > 0 && lines_processed >= max_events
                    shutdown_reason = "max_events"
                    Stream.stop!(stop_signal)
                    done = true
                    break
                end
            end
        end
    catch e
        if e isa InterruptException
            shutdown_reason = "signal"
            StructuredLog.info("shutdown signal received")
        else
            rethrow()
        end
    finally
        Stream.stop!(stop_signal)
        try; close(ch); catch; end
        try
            timedwait(() -> istaskdone(producer),
                      Float64(opts["shutdown-timeout"]))
        catch
        end
        # A producer that died on an I/O error (source removed mid-run,
        # permission loss) → documented exit 3, not a silent clean exit.
        if istaskdone(producer) && istaskfailed(producer)
            producer_failed = true
            err = try; producer.result; catch e; e; end
            StructuredLog.error_event("input source failed";
                                      error = sprint(showerror, err))
        end
        if webhook_ch !== nothing
            # Close (no stop flag) so the worker drains queued alerts,
            # bounded by shutdown-timeout; abort only if it overruns.
            try; close(webhook_ch); catch; end
            drained = try
                timedwait(() -> istaskdone(webhook_task),
                          Float64(opts["shutdown-timeout"]))
            catch
                :error
            end
            if drained !== :ok
                try; webhook_stop[] = true; catch; end
            end
        end
        if mem_ch !== nothing
            try; mem_stop[] = true; catch; end
            try; close(mem_ch); catch; end
            try
                timedwait(() -> istaskdone(mem_task),
                          Float64(opts["shutdown-timeout"]))
            catch
            end
            try
                Memory.SQLite.finalize_session!(mem_db, mem_session_id;
                    exit_code = (exit_on_trigger && triggers_total > 0) ? 1 : 0)
            catch
            end
            if mem_writes_dropped > 0 ||
               (mem_stats !== nothing && mem_stats.dropped_ops > 0)
                StructuredLog.warn("memory writes incomplete";
                    dropped_enqueue = mem_writes_dropped,
                    dropped_ops = mem_stats === nothing ? 0 : mem_stats.dropped_ops,
                    flush_errors = mem_stats === nothing ? 0 : mem_stats.flush_errors)
            end
        end
        _emit_shutdown(triggers_total, shutdown_reason)
    end

    # Exit-code contract (docs/json-schema.md):
    #   130 killed by signal during shutdown
    #   3   I/O error (source failed)
    #   1   at least one rule fired (only with --exit-on-trigger)
    #   0   clean
    shutdown_reason == "signal" && return 130
    producer_failed && return 3
    return (exit_on_trigger && triggers_total > 0) ? 1 : 0
end

using SHA: sha256

"""
    _rules_fingerprint(opts) -> String

12-hex-char SHA-256 of the rules file **content** (or the literal
`<defaults>` token), so two stream instances pointed at the same
rules share warm-state namespace — and editing the rules in place
gets a fresh namespace instead of inheriting stale counters.
"""
function _rules_fingerprint(opts)
    path = String(opts["rules"])
    payload = if isempty(path)
        Vector{UInt8}("<defaults>")
    elseif isfile(path)
        read(path)
    else
        Vector{UInt8}(path)
    end
    return bytes2hex(sha256(payload))[1:12]
end

"""
    _warn_inert_rules(rs, inferer)

Boot-time diagnostic: a rule whose metric / model can never be
served by the loaded bundle silently never fires — surface that
instead of leaving the operator to notice weeks later.
"""
function _warn_inert_rules(rs, inferer)
    for r in rs.rules
        reason = if r isa Rules.NovelClusterRule
            inferer.bundle_kind === :drain ? nothing :
                "requires a drain model (`--model drain.jld2`)"
        elseif r isa Rules.ScoreThresholdRule
            if startswith(r.metric, "transformer_decoder.") &&
               inferer.bundle_kind !== :transformer_decoder
                "requires a transformer_decoder model"
            elseif startswith(r.metric, "novelty.") &&
                   inferer.detector === nothing
                "requires --detector"
            else
                nothing
            end
        else
            nothing
        end
        reason === nothing && continue
        StructuredLog.warn("rule can never fire with the loaded models";
                           rule_id = r.id, reason = reason)
    end
    return nothing
end

# Build the WriteTrigger payload from a Rules.TriggerEvent + the
# InferResult dict and push it onto the writer channel. Non-blocking:
# a full or closed channel drops the write (counted by the caller via
# the returned Bool) rather than stalling the inference loop —
# stdout remains the durable sink.
function _enqueue_trigger_write(ch, session_id::Int, t, ir::AbstractDict)
    frame  = get(ir, "frame", nothing)
    drain  = get(ir, "drain", nothing)
    dcid   = drain isa AbstractDict ? get(drain, "cluster_id", nothing) : nothing
    sig    = _model_signals_for_ir(ir)
    payload = Dict{Symbol, Any}(
        :session_id        => session_id,
        :rule_id           => t.rule_id,
        :rule_kind         => string(t.rule_kind),
        :severity          => string(t.severity),
        :line_id           => t.line_id,
        :line              => t.line,
        :fields            => t.fields,
        :ts                => t.ts,
        :frame             => frame,
        :drain_cluster_id  => dcid,
        :model_signals     => sig,
    )
    return Memory.SQLite.try_put!(ch, Memory.SQLite.WriteTrigger(payload))
end

function _enqueue_line_write(ch, session_id::Int, ev, ir::AbstractDict)
    drain = get(ir, "drain", nothing)
    dcid  = drain isa AbstractDict ? get(drain, "cluster_id", nothing) : nothing
    sig   = _model_signals_for_ir(ir)
    payload = Dict{Symbol, Any}(
        :session_id        => session_id,
        :line_id           => ev.line_id,
        :line              => ev.line,
        :ts                => ev.ts,
        :drain_cluster_id  => dcid,
        :model_signals     => sig,
    )
    return Memory.SQLite.try_put!(ch, Memory.SQLite.WriteLine(payload))
end

# Rule ids for pattern-origin triggers are "pattern:<id>:<name>".
# Names may contain ':' so we parse only the numeric second field.
function _pattern_id_from_rule(rule_id::AbstractString)
    startswith(rule_id, "pattern:") || return nothing
    rest = SubString(rule_id, ncodeunits("pattern:") + 1)
    colon = findfirst(':', rest)
    idstr = colon === nothing ? rest : SubString(rest, 1, colon - 1)
    return tryparse(Int, idstr)
end

function _enqueue_pattern_match_write(ch, session_id::Int, pattern_id::Int, t)
    payload = Dict{Symbol, Any}(
        :session_id => session_id,
        :pattern_id => pattern_id,
        :line_id    => t.line_id,
        :ts         => t.ts,
        :evidence   => t.fields,
    )
    return Memory.SQLite.try_put!(ch, Memory.SQLite.WritePatternMatch(payload))
end

# Extract the "model signals" slice of an InferResult: every key
# whose value is itself a Dict (the per-model namespaces) plus the
# `novelty` block when present. Skip the stable top-level fields.
function _model_signals_for_ir(ir::AbstractDict)
    keep = Dict{String, Any}()
    for (k, v) in ir
        ks = String(k)
        ks in ("line", "line_id", "ts", "drain", "frame", "partial") && continue
        v isa AbstractDict || continue
        keep[ks] = v
    end
    return isempty(keep) ? nothing : keep
end

function _spawn_producer(opts, ch, stop_signal, stats)
    src = String(opts["data"])
    max_bytes = Int(opts["max-line-bytes"])
    if isempty(src) || src == "-"
        # Stdin path. The redirected `stdin` inside the test harness
        # works the same way.
        return @async begin
            try
                Stream.stream_stdin!(stdin, ch; stop = stop_signal,
                                      stats = stats,
                                      max_line_bytes = max_bytes)
            finally
                try; close(ch); catch; end
            end
        end
    elseif opts["tail"]
        return @async begin
            try
                Stream.tail_file!(src, ch; stop = stop_signal, stats = stats,
                                   max_line_bytes = max_bytes,
                                   from_start = false)
            finally
                try; close(ch); catch; end
            end
        end
    else
        return @async begin
            try
                open(src, "r") do io
                    Stream.stream_stdin!(io, ch; stop = stop_signal,
                                          stats = stats,
                                          max_line_bytes = max_bytes)
                end
            finally
                try; close(ch); catch; end
            end
        end
    end
end

function _build_inferer(opts, framed_mode::Symbol)::StreamInferer
    bundle_kind = nothing
    artifact    = nothing
    if !isempty(opts["model"])
        bundle = Persistence.load(String(opts["model"]))
        bundle_kind = bundle.kind
        artifact    = Persistence.rehydrate(bundle)
    end
    det = nothing
    if !isempty(opts["detector"])
        det = Persistence.load_and_rehydrate(String(opts["detector"]))
        det isa ValueNoveltyDetector ||
            throw(ArgumentError("--detector must be a ValueNoveltyDetector bundle"))
    end
    # Near-duplicate NLL memo: only meaningful for the (expensive)
    # transformer_decoder path, keyed by the masked template.
    memo = nothing
    if get(opts, "dedup", "off") == "masked" &&
       bundle_kind === :transformer_decoder
        memo = Dedup.LRUMemo{String, Float64}(Int(get(opts, "dedup-window", 4096)))
    end
    return StreamInferer(bundle_kind, artifact, det, framed_mode, memo)
end

"""
    _infer_cheap(inf, ev) -> (ir, needs_nll::Bool, body::String)

The per-line inference work that is *cheap*: framing, Drain template
assignment, value-novelty. The (expensive) transformer_decoder NLL is
NOT computed here — `needs_nll` flags that `ir["transformer_decoder"]`
is still missing so the caller can batch the forward across a
micro-batch of lines (see the worker loop) or fill it single-line
(see [`_infer`]). `body` is the framed message used as the model input.
"""
function _infer_cheap(inf::StreamInferer, ev::Stream.LineEvent)
    raw_line = ev.line
    body = raw_line
    frame_payload = nothing
    if inf.framed === :auto
        f = parse_frame(raw_line)
        body = String(f.message)
        frame_payload = Dict{String, Any}(
            "source"    => string(f.source),
            "host"      => String(f.host),
            "app"       => String(f.app),
            "timestamp" => String(f.timestamp),
        )
    end

    ir = Dict{String, Any}(
        "line"    => body,
        "line_id" => ev.line_id,
        "ts"      => string(ev.ts),
    )
    frame_payload === nothing || (ir["frame"] = frame_payload)
    ev.partial && (ir["partial"] = true)

    needs_nll = false
    if inf.bundle_kind === :drain
        cid, tpl = process!(inf.artifact, body)
        ir["drain"] = Dict{String, Any}("cluster_id" => cid, "template" => tpl)
    elseif inf.bundle_kind === :transformer_decoder
        needs_nll = true                     # filled by the batch forward
    end

    if inf.detector !== nothing
        _, values = mask_lines_with_values([body])
        if !isempty(values)
            nov = value_novelty(inf.detector, values[1])
            ir["novelty"] = Dict{String, Any}("value_novelty" => Float64(nov))
        end
    end
    return ir, needs_nll, body
end

"""
    _infer(inf, ev) -> Dict{String, Any}

Single-line inference (framing + drain + novelty + transformer NLL),
used by callers that don't micro-batch (e.g. `rules --dry-run`). The
streaming worker uses [`_infer_cheap`] + a batched NLL forward
instead. Stable shape per plan §3.
"""
function _infer(inf::StreamInferer, ev::Stream.LineEvent)
    ir, needs_nll, body = _infer_cheap(inf, ev)
    if needs_nll
        # Memoize by masked template when --dedup masked is on: equal
        # templates → identical masked token sequence → identical NLL.
        nll = if inf.nll_memo === nothing
            Float64(_per_line_decoder_nll(inf.artifact, [body])[1])
        else
            Dedup.memoize!(inf.nll_memo, mask_line(body),
                () -> Float64(_per_line_decoder_nll(inf.artifact, [body])[1]))
        end
        ir["transformer_decoder"] = Dict{String, Any}("nll" => nll)
    end
    return ir
end

"""
    _fill_batch_nll!(inf, buffer)

Fill `ir["transformer_decoder"]` for every buffer entry whose
`needs_nll` is set, using ONE batched transformer forward. Entries are
grouped by masked template (equal template ⇒ identical NLL), so each
distinct template is computed once and the memo (if any) is populated.
`buffer` is a Vector of mutable `[ir, needs_nll, body]` triples.
"""
function _fill_batch_nll!(inf::StreamInferer, buffer::Vector)
    inf.bundle_kind === :transformer_decoder || return
    # Group the cache-misses by masked template.
    reps = String[]                 # one representative body per template
    key_of = String[]               # masked key per rep (for the memo)
    idx_by_key = Dict{String, Int}()
    entries_by_key = Dict{String, Vector{Int}}()
    for (i, e) in enumerate(buffer)
        e[2] || continue            # needs_nll
        body = e[3]::String
        key = mask_line(body)
        j = get(idx_by_key, key, 0)
        if j == 0
            push!(reps, body); push!(key_of, key)
            idx_by_key[key] = length(reps)
            entries_by_key[key] = Int[i]
        else
            push!(entries_by_key[key], i)
        end
    end
    isempty(reps) && return
    nlls = _decoder_nll_batch(inf.artifact, reps)   # one forward for the batch
    for (r, key) in enumerate(key_of)
        nll = nlls[r]
        inf.nll_memo === nothing ||
            Dedup.memoize!(inf.nll_memo, key, () -> nll)
        for i in entries_by_key[key]
            buffer[i][1]["transformer_decoder"] = Dict{String, Any}("nll" => nll)
            buffer[i][2] = false
        end
    end
    return
end

"""
    _cheap_or_memo!(inf, ev) -> Vector{Any}  # [ir, needs_nll, body]

Run [`_infer_cheap`]; if a memo (--dedup) is present and already holds
this line's masked template, fill the NLL from the cache immediately
so it isn't sent to the batch forward. Returns a mutable 3-vector the
worker collects into a micro-batch.
"""
function _cheap_or_memo!(inf::StreamInferer, ev::Stream.LineEvent)
    ir, needs_nll, body = _infer_cheap(inf, ev)
    if needs_nll && inf.nll_memo !== nothing
        key = mask_line(body)
        if haskey(inf.nll_memo, key)
            nll = Dedup.memoize!(inf.nll_memo, key, () -> 0.0)  # hit
            ir["transformer_decoder"] = Dict{String, Any}("nll" => nll)
            needs_nll = false
        end
    end
    return Any[ir, needs_nll, body]
end

# ---------------------------------------------------------------------------
# JSON emitters — stable on-the-wire schema (see docs/json-schema.md).
# ---------------------------------------------------------------------------

_iso_now() = string(Dates.format(now(UTC),
                                  Dates.dateformat"yyyy-mm-ddTHH:MM:SS.sss"), "Z")

function _emit_line_records(ir, ev, triggers, opts)
    if opts["emit-all"]
        rec = Dict{String, Any}("event" => "line")
        for (k, v) in ir
            rec[k] = v
        end
        rec["triggered"] = [t.rule_id for t in triggers]
        println(stdout, JSON3.write(rec))
    end
    for t in triggers
        rec = Dict{String, Any}(
            "event"     => "trigger",
            "ts"        => string(t.ts),
            "rule_id"   => t.rule_id,
            "rule_kind" => string(t.rule_kind),
            "severity"  => string(t.severity),
            "line_id"   => t.line_id,
            "line"      => t.line,
            "fields"    => t.fields,
            "sinks"     => t.sinks,
        )
        if haskey(ir, "frame")
            rec["frame"] = ir["frame"]
        end
        if haskey(ir, "drain")
            rec["drain"] = ir["drain"]
        end
        if haskey(ir, "transformer_decoder")
            rec["transformer_decoder"] = ir["transformer_decoder"]
        end
        println(stdout, JSON3.write(rec))
    end
    flush(stdout)
end

function _emit_status(started_at, lines_processed, triggers_total,
                      by_rule, tail_stats, ch, inferer = nothing)
    uptime = time() - started_at
    rate = uptime > 0 ? lines_processed / uptime : 0.0
    rec = Dict{String, Any}(
        "event"   => "status",
        "ts"      => _iso_now(),
        "uptime_s" => uptime,
        "lines"   => Dict{String, Any}(
            "total"            => lines_processed,
            "rate_per_s"       => round(rate; digits = 2),
            "dropped_oversize" => tail_stats.dropped_oversize,
        ),
        "triggers" => Dict{String, Any}(
            "total"   => triggers_total,
            "by_rule" => by_rule,
        ),
        "rotations" => tail_stats.rotations,
        "queue"     => Dict{String, Any}(
            "ingest" => length(ch.data),
        ),
    )
    if inferer !== nothing && inferer.nll_memo !== nothing
        m = inferer.nll_memo
        total = m.hits + m.misses
        rec["dedup"] = Dict{String, Any}(
            "hits"      => m.hits,
            "misses"    => m.misses,
            "hit_rate"  => total > 0 ? round(m.hits / total; digits = 3) : 0.0,
            "cache_size" => length(m),
        )
    end
    println(stdout, JSON3.write(rec))
    flush(stdout)
end

function _emit_shutdown(triggers_total, reason::AbstractString = "eof")
    rec = Dict{String, Any}(
        "event"     => "shutdown",
        "ts"        => _iso_now(),
        "reason"    => reason,
        "triggers_total" => triggers_total,
    )
    try
        println(stdout, JSON3.write(rec))
        flush(stdout)
    catch
    end
end

function _trigger_to_dict(t, ir)
    return Dict{String, Any}(
        "event"     => "trigger",
        "ts"        => string(t.ts),
        "rule_id"   => t.rule_id,
        "rule_kind" => string(t.rule_kind),
        "severity"  => string(t.severity),
        "line_id"   => t.line_id,
        "line"      => t.line,
        "fields"    => t.fields,
    )
end

function _maybe_build_webhook_sink(opts)
    isempty(opts["webhook"]) && return nothing
    url = String(opts["webhook"])
    fmt = Symbol(opts["webhook-format"])
    fmt in (:raw, :slack, :alertmanager) ||
        throw(ArgumentError("--webhook-format must be raw|slack|alertmanager"))
    spec = Rules.SinkSpec("webhook:cli", :webhook, url, "POST",
                          Dict{String,String}("Content-Type" => "application/json"),
                          String(opts["webhook-secret-env"]), fmt)
    return SinksWebhook.WebhookSink(spec = spec)
end

function _print_stream_help()
    println("""
    usage: logcluster stream [--model PATH] [--detector PATH] [--rules PATH]
                             [--data FILE] [--tail] [--framed raw|auto]
                             [--emit-all] [--warmup-lines N] [--warmup-seconds S]
                             [--status-interval S] [--max-line-bytes N]
                             [--webhook URL] [--webhook-secret-env VAR]
                             [--webhook-format raw|slack|alertmanager]
                             [--quiet] [--exit-on-trigger]
                             [--shutdown-timeout S]
                             [--log-format json|text] [--log-level LVL]
                             [--max-events N]

    Long-running line-by-line inference + rule evaluation. Reads
    newline-terminated records from stdin (default) or --data, runs
    each one through the loaded model (if any), evaluates the rule
    bundle, and emits JSON-line trigger records on stdout. See
    docs/json-schema.md for the wire schema.

    Sources:
      --data FILE           file path; defaults to stdin when omitted or `-`.
      --tail                follow --data forever, honouring logrotate-style
                            rotation (inode change) and in-place truncation.

    Inference:
      --model PATH          drain | transformer_decoder bundle. Drain
                            contributes cluster_id + template; transformer
                            contributes per-line nll (perplexity proxy).
      --detector PATH       ValueNoveltyDetector bundle; adds value_novelty.
      --framed auto         strip the collector envelope (RFC 5424 / CRI /
                            Docker JSON) before scoring; the frame metadata
                            is attached to each record.

    Rules:
      --rules PATH          rules.json (see plan §2). Defaults to
                            `Rules.default_rules()` (novel template,
                            score p99, error rate spike, volume anomaly,
                            fatal keywords, OOM-killer regex).
      --warmup-lines N      buffer the first N lines before any rule with
                            warmup_required can fire (default 500).
      --warmup-seconds S    fallback wall-clock warmup (default 30 s).

    Output:
      --emit-all            emit one `event:"line"` record per input line
                            (default off — triggers + status only).
      --status-interval S   emit a `event:"status"` heartbeat every S
                            seconds (default 10 s; 0 disables).
      --webhook URL         shorthand for a webhook:cli sink overlayed
                            onto the rules; triggers route to it.
      --log-format text|json (default json). Stderr is JSON-lines by
                            default for log shippers.
      --log-level LVL       debug | info | warn | error (default info).

    Lifecycle:
      --quiet               suppress the stdout status heartbeat.
      --exit-on-trigger     return 1 if at least one rule fired.
      --shutdown-timeout S  graceful drain budget (default 5 s).
      --max-events N        stop after processing N lines (test hook).

    Long-term memory:
      --memory PATH         SQLite database for triggers + sessions.
                            The file is created + migrated to head on
                            boot. Writes go through an async batched
                            writer (no I/O on the inference path).
      --persist-lines MODE  none (default) | sampled | all. Sampled
                            keeps 1-in-N (--persist-lines-rate, default
                            100) full line records so `insights
                            --episodes` has a cluster-id stream to mine.

    Performance:
      --dedup MODE          off (default) | masked. `masked` memoizes
                            the transformer_decoder NLL by the line's
                            typed-slot template (equal templates → equal
                            masked token sequence → identical NLL), so
                            the transformer forward is skipped on
                            near-duplicate lines. Rule evaluation still
                            runs on every line — only the model forward
                            is elided. Hit-rate is reported in the status
                            heartbeat. No effect without a
                            transformer_decoder --model.
      --dedup-window N      recent-window cache capacity (default 4096).
      --batch-lines N       micro-batch the transformer forward: process
                            up to N already-queued lines in one forward
                            pass (default 1 = today's behavior). Only
                            drains the backlog — never waits when the
                            channel is empty, so light load adds no
                            latency. Per-line NLL is byte-identical to
                            the unbatched path (the decoder attends only
                            within a sequence). No effect without a
                            transformer_decoder --model.
      --batch-ms M          under bursty load, wait up to M ms for a
                            fuller batch (default 0 = no wait).
    """)
end

# ---------------------------------------------------------------------------
# rules — introspection / authoring helper.
# ---------------------------------------------------------------------------

function cmd_rules(args::Vector{String})::Int
    specs = [
        ("print-defaults", false, :bool),
        ("validate",       "",    :path),
        ("explain",        "",    :path),
        ("dry-run",        false, :bool),
        ("rules",          "",    :path),
        ("data",           "-",   :path),
        ("model",          "",    :path),
        ("warmup-lines",   0,     :int),
        ("warmup-seconds", 0.0,   :float),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_rules_help(); return 0)

    if opts["print-defaults"]
        path = joinpath(@__DIR__, "rules", "defaults.json")
        print(stdout, read(path, String))
        return 0
    end

    if !isempty(opts["validate"])
        try
            Rules.load_rules(String(opts["validate"]))
            println(stderr, "rules: OK")
            return 0
        catch e
            println(stderr, "rules: ", sprint(showerror, e))
            return 2
        end
    end

    if !isempty(opts["explain"])
        rs = Rules.load_rules(String(opts["explain"]))
        for r in rs.rules
            println(stdout, rpad(r.id, 24),
                    "  kind=", Rules._kind_symbol(r),
                    "  severity=", r.severity,
                    "  cooldown_s=", r.cooldown_s)
        end
        return 0
    end

    if opts["dry-run"]
        isempty(opts["rules"]) &&
            throw(ArgumentError("--dry-run needs --rules FILE"))
        rs = Rules.load_rules(String(opts["rules"]);
                              warmup_lines   = Int(opts["warmup-lines"]),
                              warmup_seconds = Float64(opts["warmup-seconds"]))
        inferer = _build_inferer(Dict("model"    => opts["model"],
                                       "detector" => "",
                                       "framed"   => "raw"), :raw)
        lines = read_lines(opts["data"])
        by_rule = Dict{String, Int}()
        for (i, l) in enumerate(lines)
            ev = Stream.LineEvent(String(l), Int(i), now(UTC); partial = false)
            ir = _infer(inferer, ev)
            for t in Rules.evaluate(rs, ir)
                by_rule[t.rule_id] = get(by_rule, t.rule_id, 0) + 1
            end
        end
        println(stdout, JSON3.write(Dict("lines" => length(lines),
                                          "by_rule" => by_rule)))
        return 0
    end

    _print_rules_help()
    return 0
end

function _print_rules_help()
    println("""
    usage: logcluster rules --print-defaults
           logcluster rules --validate FILE
           logcluster rules --explain  FILE
           logcluster rules --dry-run --rules FILE --data FILE [--model M]

    Introspect / author rules bundles.

    --print-defaults    dump the bundled `Rules.default_rules()` JSON
                        to stdout (handy as a starting template).
    --validate FILE     parse + type-check; exits 0 on success, 2 on
                        schema problems.
    --explain  FILE     print each rule's resolved severity, cooldown,
                        and dispatch kind.
    --dry-run           replay --data through the rule engine offline
                        and print `{lines, by_rule}` JSON, so you can
                        tune thresholds before going live. Optional
                        --model adds drain / transformer signals so
                        score_threshold + novel_cluster rules see real
                        data.
    """)
end

# ---------------------------------------------------------------------------
# patterns — list / add / enable / disable / delete user-curated patterns.
# ---------------------------------------------------------------------------

function cmd_patterns(args::Vector{String})::Int
    isempty(args) && (_print_patterns_help(); return 0)
    sub = args[1]
    rest = collect(String, args[2:end])
    if sub == "--help" || sub == "-h" || sub == "help"
        _print_patterns_help(); return 0
    elseif sub == "list"
        return _cmd_patterns_list(rest)
    elseif sub == "add"
        return _cmd_patterns_add(rest)
    elseif sub == "enable" || sub == "disable"
        return _cmd_patterns_toggle(rest, sub == "enable")
    elseif sub == "delete" || sub == "rm"
        return _cmd_patterns_delete(rest)
    elseif sub == "explain"
        return _cmd_patterns_explain(rest)
    else
        println(stderr, "logcluster patterns: unknown subcommand `$sub`")
        return 2
    end
end

function _cmd_patterns_list(args::Vector{String})::Int
    specs = [("memory", "", :path), ("all", false, :bool), ("json", false, :bool)]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    rows = Memory.PatternCatalog.list(db; enabled_only = !opts["all"])
    if opts["json"]
        for p in rows
            println(stdout, JSON3.write(Dict(
                "id"             => p.id,
                "name"           => p.name,
                "description"    => p.description,
                "severity"       => String(p.severity),
                "cooldown_s"     => p.cooldown_s,
                "match_kind"     => String(p.match_kind),
                "drain_template" => p.match_drain_template_id,
                "regex"          => p.match_regex === nothing ? nothing :
                                    string(p.match_regex.pattern),
                "keywords"       => p.match_keywords,
                "enabled"        => p.enabled,
            )))
        end
    else
        println(stdout, "id\tname\tkind\tseverity\tenabled\tdetail")
        for p in rows
            detail = if p.match_kind === :drain
                "template=$(p.match_drain_template_id)"
            elseif p.match_kind === :regex
                "regex=$(string(p.match_regex.pattern))"
            else
                "keywords=" * join(p.match_keywords, ",")
            end
            println(stdout, p.id, '\t', p.name, '\t', p.match_kind, '\t',
                    p.severity, '\t', p.enabled ? "yes" : "no", '\t', detail)
        end
    end
    return 0
end

function _cmd_patterns_add(args::Vector{String})::Int
    specs = [
        ("memory",        "",   :path),
        ("name",          "",   :string),
        ("description",   "",   :string),
        ("severity",      "warn", :string),
        ("cooldown-s",    60.0, :float),
        ("from-trigger",  0,    :int),
        ("kind",          "",   :string),    # keyword | regex | drain
        ("regex",         "",   :string),
        ("keyword",       "",   :string),    # repeatable not supported; comma-separated
        ("drain-template",0,    :int),
        ("case-sensitive", false, :bool),
        ("created-by",    "",   :string),
    ]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    isempty(opts["name"])   && throw(ArgumentError("--name is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)

    pid = if opts["from-trigger"] > 0
        mk = isempty(opts["kind"]) ? :drain : Symbol(opts["kind"])
        Memory.PatternCatalog.pin_from_trigger(db, Int(opts["from-trigger"]);
            name = String(opts["name"]),
            description = String(opts["description"]),
            severity = Symbol(opts["severity"]),
            cooldown_s = Float64(opts["cooldown-s"]),
            match_kind = mk,
            case_sensitive = opts["case-sensitive"],
            created_by = String(opts["created-by"]))
    else
        kind = Symbol(opts["kind"])
        kind in (:keyword, :regex, :drain) ||
            throw(ArgumentError("--kind must be keyword | regex | drain"))
        # Drop empty tokens ("a,,b" / trailing comma) — an empty
        # keyword would `occursin("", line)`-match every line.
        kws = isempty(opts["keyword"]) ? String[] :
              String[String(strip(k)) for k in split(String(opts["keyword"]), ',')
                     if !isempty(strip(k))]
        kind === :keyword && isempty(kws) &&
            throw(ArgumentError("--kind keyword needs at least one non-empty --keyword"))
        Memory.PatternCatalog.pin_manual(db;
            name = String(opts["name"]),
            description = String(opts["description"]),
            severity = Symbol(opts["severity"]),
            cooldown_s = Float64(opts["cooldown-s"]),
            match_kind = kind,
            match_drain_template_id = opts["drain-template"] > 0 ?
                Int(opts["drain-template"]) : nothing,
            match_regex = String(opts["regex"]),
            match_keywords = kws,
            case_sensitive = opts["case-sensitive"],
            created_by = String(opts["created-by"]))
    end
    println(stdout, JSON3.write(Dict("id" => pid, "name" => String(opts["name"]))))
    return 0
end

function _cmd_patterns_toggle(args::Vector{String}, on::Bool)::Int
    specs = [("memory", "", :path), ("id", 0, :int)]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    opts["id"] > 0          || throw(ArgumentError("--id is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    Memory.PatternCatalog.enable!(db, Int(opts["id"]), on)
    return 0
end

function _cmd_patterns_delete(args::Vector{String})::Int
    specs = [("memory", "", :path), ("id", 0, :int)]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    opts["id"] > 0          || throw(ArgumentError("--id is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    Memory.PatternCatalog.delete!(db, Int(opts["id"]))
    return 0
end

function _cmd_patterns_explain(args::Vector{String})::Int
    specs = [("memory", "", :path), ("id", 0, :int)]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    opts["id"] > 0          || throw(ArgumentError("--id is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    p = Memory.PatternCatalog.get_pattern(db, Int(opts["id"]))
    p === nothing && (println(stderr, "pattern not found"); return 2)
    println(stdout, JSON3.write(Dict(
        "id"          => p.id,
        "name"        => p.name,
        "description" => p.description,
        "severity"    => String(p.severity),
        "cooldown_s"  => p.cooldown_s,
        "match_kind"  => String(p.match_kind),
        "drain_template" => p.match_drain_template_id,
        "regex"       => p.match_regex === nothing ? nothing : string(p.match_regex.pattern),
        "keywords"    => p.match_keywords,
        "enabled"     => p.enabled,
        "created_at"  => p.created_at,
        "created_by"  => p.created_by,
    )))
    return 0
end

function _print_patterns_help()
    println("""
    usage: logcluster patterns list    --memory PATH [--all] [--json]
           logcluster patterns add     --memory PATH --name N --kind K [...]
           logcluster patterns add     --memory PATH --name N --from-trigger ID
           logcluster patterns enable  --memory PATH --id N
           logcluster patterns disable --memory PATH --id N
           logcluster patterns delete  --memory PATH --id N
           logcluster patterns explain --memory PATH --id N

    Author / inspect user-curated patterns persisted in the SQLite
    memory store. Patterns auto-promote into rules on the next
    `stream` run when --use-patterns is set; drain-template patterns
    are matched via an exact-cluster-id sidecar pass.

    --kind selects the matcher: keyword | regex | drain.
    --keyword "a,b,c"  comma-separated list of keywords (case insensitive
                        by default; pass --case-sensitive to opt in).
    --regex   "PAT"     literal regex.
    --drain-template N  Drain-cluster id for kind = drain.

    --from-trigger ID   pin a pattern from an existing trigger row.
                        Uses the trigger's drain_cluster_id (default)
                        or its raw line, depending on --kind.
    """)
end

# ---------------------------------------------------------------------------
# query / insights / report — read-only views over the SQLite store.
# ---------------------------------------------------------------------------

function cmd_query(args::Vector{String})::Int
    isempty(args) && (_print_query_help(); return 0)
    sub = args[1]
    rest = collect(String, args[2:end])
    sub == "--help" && (_print_query_help(); return 0)
    if sub == "triggers"
        return _cmd_query_triggers(rest)
    elseif sub == "sessions"
        return _cmd_query_sessions(rest)
    elseif sub == "patterns"
        return _cmd_query_patterns(rest)
    else
        println(stderr, "logcluster query: unknown subcommand `$sub`"); return 2
    end
end

function _cmd_query_triggers(args::Vector{String})::Int
    specs = [
        ("memory",  "",     :path),
        ("since",   "24h",  :string),
        ("until",   "",     :string),
        ("rule",    "",     :string),
        ("severity", "",    :string),
        ("cluster", 0,      :int),
        ("limit",   50,     :int),
        ("json",    false,  :bool),
    ]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    since = Memory.SQLite.epoch_ms_since(String(opts["since"]))
    until = isempty(opts["until"]) ? nothing :
            Memory.SQLite.epoch_ms_since(String(opts["until"]))
    rows = Memory.SQLite.triggers(db;
        since = since, until = until,
        rule_id = isempty(opts["rule"]) ? nothing : String(opts["rule"]),
        severity = isempty(opts["severity"]) ? nothing : String(opts["severity"]),
        cluster_id = opts["cluster"] > 0 ? Int(opts["cluster"]) : nothing,
        limit = Int(opts["limit"]))
    _emit_rows(rows, opts["json"];
        cols = (:id, :ts, :rule_id, :severity, :line_id, :line))
    return 0
end

function _cmd_query_sessions(args::Vector{String})::Int
    specs = [
        ("memory", "",    :path),
        ("since",  "30d", :string),
        ("limit",  50,    :int),
        ("json",   false, :bool),
    ]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    # sessions.started_at is ISO-8601 TEXT, so compare against an ISO
    # cutoff (not the epoch-ms integer, which sorts lexicographically
    # wrong against ISO text and made --since a silent no-op).
    since_iso = Memory.SQLite.iso_from_epoch_ms(
        Memory.SQLite.epoch_ms_since(String(opts["since"])))
    rows = Memory.SQLite._rows(db,
        "SELECT id, started_at, ended_at, host, model_path, rules_path, " *
        "exit_code FROM sessions WHERE started_at >= ? " *
        "ORDER BY id DESC LIMIT ?",
        (since_iso, Int(opts["limit"])))
    _emit_rows(rows, opts["json"];
        cols = (:id, :started_at, :ended_at, :host, :exit_code))
    return 0
end

function _cmd_query_patterns(args::Vector{String})::Int
    specs = [("memory", "", :path), ("json", false, :bool)]
    opts = parse_flags(args, specs)
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    rows = Memory.PatternCatalog.list(db; enabled_only = false)
    if opts["json"]
        for p in rows
            println(stdout, JSON3.write(Dict("id" => p.id, "name" => p.name,
                "kind" => String(p.match_kind), "enabled" => p.enabled,
                "severity" => String(p.severity))))
        end
    else
        println(stdout, "id\tname\tkind\tseverity\tenabled")
        for p in rows
            println(stdout, p.id, '\t', p.name, '\t', p.match_kind, '\t',
                    p.severity, '\t', p.enabled ? "yes" : "no")
        end
    end
    return 0
end

function _emit_rows(rows, json::Bool; cols::Tuple)
    if json
        for r in rows
            println(stdout, JSON3.write(_row_to_dict(r)))
        end
    else
        println(stdout, join(String.(cols), '\t'))
        for r in rows
            vals = [string(get(r, c, "")) for c in cols]
            println(stdout, join(vals, '\t'))
        end
    end
end

_row_to_dict(r::NamedTuple) =
    Dict{String, Any}(String(k) => v for (k, v) in pairs(r))

function _print_query_help()
    println("""
    usage: logcluster query triggers --memory PATH [--since 24h] [--rule X]
                                     [--severity S] [--cluster N] [--limit 50]
                                     [--json]
           logcluster query sessions  --memory PATH [--since 30d] [--limit 50]
           logcluster query patterns  --memory PATH [--json]

    Read-only SQL-backed views over the long-term store.

    --since DUR | ISO     accepts 24h, 7d, 30m, 45s, or an ISO-8601 timestamp.
    --json                emit NDJSON instead of TSV.
    """)
end

function cmd_insights(args::Vector{String})::Int
    specs = [
        ("memory",   "",   :path),
        ("since",    "24h", :string),
        ("until",    "",    :string),
        ("limit",    20,    :int),
        ("top-rules",      false, :bool),
        ("novel",          false, :bool),
        ("burstiness",     "",    :string),    # rule_id
        ("cluster",        0,     :int),
        ("transitions",    false, :bool),
        ("episodes",       false, :bool),
        ("pinned",         false, :bool),
        ("bucket-s",       60,    :int),
        ("min-sup",        3,     :int),
        ("max-gap",        20,    :int),
        ("max-dur",        50,    :int),
        ("json",           false, :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_insights_help(); return 0)
    _apply_config!(opts, "insights")
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    since = Memory.SQLite.epoch_ms_since(String(opts["since"]))
    until = isempty(opts["until"]) ? nothing :
            Memory.SQLite.epoch_ms_since(String(opts["until"]))

    result = Dict{String, Any}()
    if opts["top-rules"]
        result["top_rules"] = Memory.Insights.top_rules_window(db;
            since = since, until = until, limit = Int(opts["limit"]))
    end
    if opts["novel"]
        result["novel"] = Memory.Insights.novel_clusters_window(db;
            since = since, until = until)
    end
    if !isempty(opts["burstiness"])
        result["burstiness"] = Memory.Insights.burstiness(db;
            rule_id = String(opts["burstiness"]),
            since = since, until = until,
            bucket_s = Int(opts["bucket-s"]))
    end
    if opts["cluster"] > 0
        result["cluster"] = Memory.Insights.cluster_view(db;
            cluster_id = Int(opts["cluster"]),
            since = since, until = until,
            bucket_s = Int(opts["bucket-s"]))
    end
    if opts["transitions"]
        result["transitions"] = Memory.Insights.transitions(db;
            since = since, until = until, top = Int(opts["limit"]))
    end
    if opts["episodes"]
        result["episodes"] = Memory.Insights.episodes(db;
            since = since, until = until,
            min_sup = Int(opts["min-sup"]),
            max_gap = Int(opts["max-gap"]),
            max_time_duration = Int(opts["max-dur"]),
            top = Int(opts["limit"]))
    end
    if opts["pinned"]
        result["pinned"] = Memory.Insights.pinned_summary(db;
            since = since, until = until)
    end

    if isempty(result)
        _print_insights_help()
        return 0
    end
    if opts["json"]
        println(stdout, JSON3.write(_jsonable(result)))
    else
        _render_insights_text(result)
    end
    return 0
end

# Convert a NamedTuple-keyed result tree into something JSON3 can
# serialise without choking on Missing.
function _jsonable(x)
    if x isa AbstractDict
        return Dict{String, Any}(String(k) => _jsonable(v) for (k, v) in x)
    elseif x isa AbstractVector
        return Any[_jsonable(v) for v in x]
    elseif x isa NamedTuple
        return Dict{String, Any}(String(k) => _jsonable(v) for (k, v) in pairs(x))
    elseif x isa Missing
        return nothing
    else
        return x
    end
end

function _render_insights_text(result::AbstractDict)
    for (section, rows) in result
        println(stdout, "## ", section)
        if isempty(rows)
            println(stdout, "  (no rows)")
            continue
        end
        first_row = rows[1]
        cols = first_row isa NamedTuple ? collect(keys(first_row)) : Symbol[]
        if !isempty(cols)
            println(stdout, "  ", join(String.(cols), '\t'))
        end
        for r in rows
            if r isa NamedTuple
                vals = [string(getproperty(r, c)) for c in cols]
                println(stdout, "  ", join(vals, '\t'))
            else
                println(stdout, "  ", r)
            end
        end
        println(stdout)
    end
end

function _print_insights_help()
    println("""
    usage: logcluster insights --memory PATH [--since 24h] [--until ISO]
                               [--top-rules] [--novel]
                               [--burstiness RULE] [--cluster N]
                               [--transitions] [--episodes] [--pinned]
                               [--limit 20] [--bucket-s 60]
                               [--min-sup 3] [--max-gap 20] [--max-dur 50]
                               [--json]

    Compose any number of view flags into one analysis run. Each
    flag adds a section to the output keyed by name. With --json,
    the output is a single Dict whose keys are the requested views.

    Views:
      --top-rules        rule_id counts in the window, by severity.
      --novel            Drain clusters first seen in the window.
      --burstiness R     bucketed trigger counts for rule R.
      --cluster N        bucketed trigger counts for drain cluster N.
      --transitions      top-K (from -> to) drain transitions.
      --episodes         top-K episodes mined over the cluster sequence.
      --pinned           per-pattern match counts (curated patterns).

    --since DUR | ISO    24h, 7d, 30m, 45s, or an ISO-8601 timestamp.
    --bucket-s N         bucket width for burstiness / cluster views.
    --min-sup / --max-gap / --max-dur tune Mining.Episodes.mv_span.
    """)
end

function cmd_report(args::Vector{String})::Int
    specs = [
        ("memory",  "",    :path),
        ("since",   "24h", :string),
        ("until",   "",    :string),
        ("format",  "md",  :string),     # md | json
        ("topk",    10,    :int),
        ("min-sup", 3,     :int),
        ("max-gap", 20,    :int),
        ("max-dur", 50,    :int),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_report_help(); return 0)
    _apply_config!(opts, "report")
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    since = Memory.SQLite.epoch_ms_since(String(opts["since"]))
    until = isempty(opts["until"]) ? nothing :
            Memory.SQLite.epoch_ms_since(String(opts["until"]))

    bundle = Dict{String, Any}(
        "since"  => opts["since"],
        "until"  => isempty(opts["until"]) ? "now" : opts["until"],
        "top_rules" => Memory.Insights.top_rules_window(db;
            since = since, until = until, limit = Int(opts["topk"])),
        "novel"     => Memory.Insights.novel_clusters_window(db;
            since = since, until = until),
        "transitions" => Memory.Insights.transitions(db;
            since = since, until = until, top = Int(opts["topk"])),
        "episodes"    => Memory.Insights.episodes(db;
            since = since, until = until,
            min_sup = Int(opts["min-sup"]), max_gap = Int(opts["max-gap"]),
            max_time_duration = Int(opts["max-dur"]),
            top = Int(opts["topk"])),
        "pinned"      => Memory.Insights.pinned_summary(db;
            since = since, until = until),
    )

    if opts["format"] == "json"
        println(stdout, JSON3.write(_jsonable(bundle)))
    else
        print(stdout, _render_report_md(bundle))
    end
    return 0
end

function _render_report_md(b::AbstractDict)
    io = IOBuffer()
    println(io, "# LogClustering insights report")
    println(io)
    println(io, "Window: ", b["since"], " → ", b["until"])
    println(io)
    println(io, "## Top rules")
    println(io)
    if isempty(b["top_rules"])
        println(io, "_no triggers in window_")
    else
        println(io, "| rule_id | severity | count |")
        println(io, "|---------|----------|-------|")
        for r in b["top_rules"]
            println(io, "| ", r.rule_id, " | ", r.severity, " | ", r.n, " |")
        end
    end
    println(io)

    println(io, "## Novel templates")
    println(io)
    if isempty(b["novel"])
        println(io, "_none_")
    else
        println(io, "| cluster_id | first_seen_ms |")
        println(io, "|------------|---------------|")
        for r in b["novel"]
            println(io, "| ", r.cluster_id, " | ", r.first_seen_ms, " |")
        end
    end
    println(io)

    println(io, "## Top cluster transitions")
    println(io)
    if isempty(b["transitions"])
        println(io, "_none_")
    else
        println(io, "| from | to | count |")
        println(io, "|------|----|-------|")
        for r in b["transitions"]
            println(io, "| ", r.from, " | ", r.to, " | ", r.n, " |")
        end
    end
    println(io)

    println(io, "## Episodes")
    println(io)
    if isempty(b["episodes"])
        println(io, "_no episodes mined_")
    else
        println(io, "| pattern | support |")
        println(io, "|---------|---------|")
        for r in b["episodes"]
            println(io, "| ", join(r.pattern, ","), " | ", r.support, " |")
        end
    end
    println(io)

    println(io, "## Pinned-pattern activity")
    println(io)
    if isempty(b["pinned"])
        println(io, "_no patterns pinned_")
    else
        println(io, "| pattern | severity | matches |")
        println(io, "|---------|----------|---------|")
        for r in b["pinned"]
            println(io, "| ", r.name, " | ", r.severity, " | ", r.n, " |")
        end
    end
    println(io)
    return String(take!(io))
end

function _print_report_help()
    println("""
    usage: logcluster report --memory PATH [--since 24h] [--until ISO]
                             [--format md|json] [--topk 10]
                             [--min-sup 3] [--max-gap 20] [--max-dur 50]

    Bundled report (Markdown by default, JSON with --format json)
    over the past window. Sections: top rules, novel templates, top
    cluster transitions, mined episodes, pinned-pattern activity.

    Drop into a runbook with:
      logcluster report --since 24h --memory /var/lib/logcluster/triggers.sqlite \\
        > /var/log/logcluster/daily.md
    """)
end

# ---------------------------------------------------------------------------
# top — TUI dashboard.
# ---------------------------------------------------------------------------

function cmd_top(args::Vector{String})::Int
    specs = [
        ("memory",    "",    :path),
        ("since",     "24h", :string),
        ("refresh-s", 1.0,   :float),
        ("once",      false, :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_top_subhelp(); return 0)
    _apply_config!(opts, "top")
    isempty(opts["memory"]) && throw(ArgumentError("--memory is required"))
    db = Memory.SQLite.open_db(String(opts["memory"]); create = false)
    Memory.SQLite.migrate!(db)
    if opts["once"] || !(stdout isa Base.TTY)
        # Test / pipe path: render one frame and return.
        print(stdout, TUI.render(db;
            since = String(opts["since"]), colour = false))
        return 0
    end
    return TUI.run(db; since = String(opts["since"]),
                                  refresh_s = Float64(opts["refresh-s"]))
end

function _print_top_subhelp()
    println("""
    usage: logcluster top --memory PATH [--since 24h] [--refresh-s 1.0]
                          [--once]

    htop-style live dashboard against the SQLite store. Refreshes
    every --refresh-s seconds (default 1). Use --once for a
    one-shot snapshot (also the default when stdout isn't a TTY,
    so the command is safe in scripts).

    Ctrl-C exits.
    """)
end

# ---------------------------------------------------------------------------
# init — bootstrap config + db + sample rules.
# ---------------------------------------------------------------------------

function cmd_init(args::Vector{String})::Int
    specs = [
        ("prefix", "", :path),
        ("force",  false, :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_init_help(); return 0)
    prefix = isempty(opts["prefix"]) ?
             dirname(Config.default_path()) :
             String(opts["prefix"])
    mkpath(prefix)
    # Data (SQLite db) lives under XDG_DATA_HOME, config under prefix.
    home = try; homedir(); catch; "."; end
    data_dir = joinpath(get(ENV, "XDG_DATA_HOME",
                            joinpath(home, ".local", "share")),
                        "logcluster")
    mkpath(data_dir)

    cfg_path   = joinpath(prefix, "config.toml")
    rules_path = joinpath(prefix, "rules.json")
    db_path    = joinpath(data_dir, "triggers.sqlite")

    created = String[]
    if opts["force"] || !isfile(cfg_path)
        _write_default_config(cfg_path, rules_path, db_path)
        push!(created, cfg_path)
    end
    if opts["force"] || !isfile(rules_path)
        src = joinpath(@__DIR__, "rules", "defaults.json")
        cp(src, rules_path; force = true)
        push!(created, rules_path)
    end
    if opts["force"] || !isfile(db_path)
        db = Memory.SQLite.open_db(db_path)
        Memory.SQLite.migrate!(db)
        push!(created, db_path)
    end

    println(stdout, JSON3.write(Dict(
        "prefix" => prefix,
        "config" => cfg_path,
        "rules"  => rules_path,
        "db"     => db_path,
        "created" => created,
    )))
    return 0
end

function _write_default_config(cfg_path::AbstractString,
                                rules_path::AbstractString,
                                db_path::AbstractString)
    body = """
    # logcluster — generated by `logcluster init`. Edit freely.
    #
    # Each section is a subcommand name; keys mirror the long flags.
    # CLI flags take priority over the config file.

    [stream]
    rules           = "$rules_path"
    memory          = "$db_path"
    framed          = "raw"
    warmup-lines    = 500
    warmup-seconds  = 30
    status-interval = 10
    log-format      = "json"
    log-level       = "info"

    [insights]
    memory = "$db_path"
    since  = "24h"

    [report]
    memory = "$db_path"
    since  = "24h"
    format = "md"
    """
    open(cfg_path, "w") do io
        write(io, body)
    end
    return nothing
end

function _print_init_help()
    println("""
    usage: logcluster init [--prefix DIR] [--force]

    Bootstrap a fresh logcluster install: a sample config.toml, a
    copy of the bundled default rules, and an empty SQLite store
    migrated to head.

    Without --prefix the config file lands at
    \$LOGCLUSTERING_CONFIG | \$XDG_CONFIG_HOME/logcluster/config.toml |
    ~/.config/logcluster/config.toml. The SQLite db lands under
    \$XDG_DATA_HOME/logcluster/triggers.sqlite.
    """)
end

# ---------------------------------------------------------------------------
# doctor — verify config + model + db + (optional) Redis.
# ---------------------------------------------------------------------------

function cmd_doctor(args::Vector{String})::Int
    specs = [
        ("config",       "",  :path),
        ("memory",       "",  :path),
        ("memory-warm",  "",  :string),
        ("rules",        "",  :path),
        ("model",        "",  :path),
        ("json",         false, :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_doctor_help(); return 0)

    findings = Vector{Tuple{String, Symbol, String}}()
    add!(label, status, detail) = push!(findings, (label, status, detail))

    # Config file.
    cfg_path = isempty(opts["config"]) ?
               Config.default_path() :
               String(opts["config"])
    if isfile(cfg_path)
        try
            cfg = Config.load(cfg_path)
            add!("config", :ok, "$cfg_path ($(length(cfg)) sections)")
        catch e
            add!("config", :fail, sprint(showerror, e))
        end
    else
        add!("config", :warn, "no config file at $cfg_path (run `logcluster init`)")
    end

    # SQLite store.
    db_path = String(opts["memory"])
    if isempty(db_path)
        add!("memory", :warn, "no --memory supplied")
    elseif !isfile(db_path)
        add!("memory", :fail, "$db_path: does not exist")
    else
        try
            db = Memory.SQLite.open_db(db_path; create = false)
            v = Memory.SQLite.migrate!(db)
            add!("memory", :ok, "$db_path (schema v$v)")
        catch e
            add!("memory", :fail, sprint(showerror, e))
        end
    end

    # Rules.
    if !isempty(opts["rules"])
        try
            Rules.load_rules(String(opts["rules"]))
            add!("rules", :ok, opts["rules"])
        catch e
            add!("rules", :fail, sprint(showerror, e))
        end
    else
        add!("rules", :warn, "no --rules supplied; defaults assumed")
    end

    # Model bundle.
    if !isempty(opts["model"])
        if !isfile(opts["model"])
            add!("model", :fail, "$(opts["model"]): does not exist")
        else
            try
                bundle = Persistence.load(String(opts["model"]))
                add!("model", :ok, "$(opts["model"]) (kind=$(bundle.kind))")
            catch e
                add!("model", :fail, sprint(showerror, e))
            end
        end
    end

    # Redis (optional).
    if !isempty(opts["memory-warm"])
        try
            store = Memory.WarmStore.open_warm(String(opts["memory-warm"]))
            if Memory.WarmStore.healthy(store)
                add!("memory-warm", :ok, opts["memory-warm"])
            else
                add!("memory-warm", :fail, "ping failed")
            end
            try; close(store); catch; end
        catch e
            add!("memory-warm", :fail, sprint(showerror, e))
        end
    end

    if opts["json"]
        out = Any[]
        for (label, status, detail) in findings
            push!(out, Dict("label" => label, "status" => String(status),
                            "detail" => detail))
        end
        println(stdout, JSON3.write(out))
    else
        for (label, status, detail) in findings
            badge = status === :ok   ? "ok  " :
                    status === :warn ? "warn" :
                                       "FAIL"
            println(stdout, badge, "  ", rpad(label, 13), "  ", detail)
        end
    end

    any(f -> f[2] === :fail, findings) ? 3 : 0
end

function _print_doctor_help()
    println("""
    usage: logcluster doctor [--config PATH] [--memory PATH]
                             [--memory-warm URL] [--rules PATH]
                             [--model PATH] [--json]

    Lightweight diagnostic: confirms the config file parses, the
    SQLite store opens + is migrated to head, the rules file
    validates, the model bundle loads, and (when supplied) Redis is
    reachable. Exits 0 when nothing is broken, 3 otherwise.
    """)
end

const SUBCOMMANDS = Dict{String, Function}(
    "train"            => cmd_train,
    "classify"         => cmd_classify,
    "score"            => cmd_score,
    "stream"           => cmd_stream,
    "rules"            => cmd_rules,
    "patterns"         => cmd_patterns,
    "query"            => cmd_query,
    "insights"         => cmd_insights,
    "report"           => cmd_report,
    "top"              => cmd_top,
    "init"             => cmd_init,
    "doctor"           => cmd_doctor,
    "select-model"     => cmd_select,
    "rca"              => cmd_rca,
    "mask"             => cmd_mask,
    "benchmark"        => cmd_benchmark,
    "download-loghub"  => cmd_download,
)

end # module CLI
