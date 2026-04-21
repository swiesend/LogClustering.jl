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
using ..Featurise: Featurise, Vocabulary, build_vocab, bow,
                   sequence_matrix, tokenise_ids
using ..Persistence
using ..PersistenceGlue
using ..AutoTune
using ..DeepKATE: DeepKATE, deep_kate
using ..VQVAE: vq_vae, assign_codes, VectorQuantizer
using ..SeqLSTM: seq_lstm, seq_lstm_loss
using ..Instance: ValueNoveltyDetector, update!, anomaly_score, combined_anomaly,
                  value_novelty
using ..Sparsity: sparsity_clusters
using ..Framing: parse_frame, SOURCE_RAW
using Lux
using Zygote
using JSON3
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
        ("quiet",     false,    :bool),
    ]
    opts = parse_flags(args, specs)
    get(opts, "help", false) && (_print_train_help(); return 0)
    isempty(opts["out"]) && throw(ArgumentError("--out is required"))
    lines = read_lines(opts["data"])
    kind = Symbol(opts["kind"])
    rng = MersenneTwister(Int(opts["seed"]))

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
    else
        throw(ArgumentError("unknown kind `$(opts["kind"])`"))
    end
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
                            [--quiet]

    Fit and persist a model.

    KIND:
      drain       streaming log-template parser (no featurisation needed)
      deep_kate   Lux autoencoder on a log-normalised BoW matrix
      vq_vae      Lux codebook AE on a binary BoW matrix
      seq_lstm    Lux next-event LSTM on padded token-id sequences

    All AE/LSTM kinds build a typed-slot-masked vocabulary on the
    training corpus and persist it in the JLD2 bundle, so a later
    `classify` pass can re-featurise new lines identically.

    --auto        run AutoTune.fit_hyperparams on the featurised corpus.
    --budget N    >0 triggers random search around the heuristic seed.
    --epochs N    training epochs (default 20).
    --batch N     mini-batch size (default 64).
    --lr F        SGD learning rate (default 0.05).
    --seed N      RNG seed (default 0).
    --seqlen N    sequence length for seq_lstm (default 16).
    --max-vocab N cap the vocabulary (default 5000).
    --min-count N minimum token frequency to keep (default 1).
    --quiet       suppress per-epoch loss lines.
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
    else
        throw(ArgumentError(
            "model kind $(bundle.kind) not yet handled by classify"))
    end
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
    X = bow(bodies, vocab; normalise = :l1)
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
    X = bow(bodies, vocab; normalise = :binary)
    codes = assign_codes(art.model, art.ps, art.st, X)
    labels = ["code-$(c)" for c in codes]
    _emit_classify(opts["out"], opts["format"], bodies, codes, labels)
    return 0
end

function _classify_seq_lstm(art, bundle, bodies, opts)
    vocab = _require_vocab(art, :seq_lstm)
    seqlen = art.seqlen === nothing ? 16 : Int(art.seqlen)
    S = sequence_matrix(bodies, vocab; seqlen = seqlen)
    # seq_lstm output is (vocab_size, batch) logits — argmax = predicted
    # next token's id. We emit that as the "cluster id"; the template
    # is the predicted token string.
    logits, _ = art.model(S, art.ps, Lux.testmode(art.st))
    ids = Int[argmax(@view logits[:, j]) for j in axes(logits, 2)]
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
        push!(scores, Float64(value_novelty(det, vs)))
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

const SUBCOMMANDS = Dict{String, Function}(
    "train"            => cmd_train,
    "classify"         => cmd_classify,
    "score"            => cmd_score,
    "mask"             => cmd_mask,
    "benchmark"        => cmd_benchmark,
    "download-loghub"  => cmd_download,
)

end # module CLI
