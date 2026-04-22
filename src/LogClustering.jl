module LogClustering

include("KATE.jl")
using .KATE

include("DeepKATE.jl")
using .DeepKATE

include("PreProc/Framing.jl")
using .Framing

include("Rust.jl")
using .Rust

include("PreProc/Masking.jl")
using .Masking

include("PreProc/Dedup.jl")
using .Dedup

include("Featurise.jl")
using .Featurise

include("Mining/Episodes.jl")
using .Episodes

include("Mining/Baselines.jl")
using .Baselines

include("Anomaly/Instance.jl")
using .Instance

include("Models/SeqLSTM.jl")
using .SeqLSTM

include("Anomaly/Sequence.jl")
using .Sequence

include("Models/VQVAE.jl")
using .VQVAE

include("Models/SimCSE.jl")
using .SimCSE

include("Models/DenoisingAE.jl")
using .DenoisingAE

include("Eval/Metrics.jl")
using .Metrics

include("Eval/Compression.jl")
using .Compression

include("Eval/Harness.jl")
using .Harness

include("Eval/CV.jl")
using .CV

include("Parsers/Drain.jl")
using .Drain3

include("PostProc/Canonical.jl")
using .Canonical

include("PostProc/Purity.jl")
using .Purity

include("PostProc/Typing.jl")
using .Typing

include("PostProc/Merge.jl")
using .Merge

include("Persistence.jl")
using .Persistence

include("PersistenceGlue.jl")
using .PersistenceGlue

include("Cluster/Sparsity.jl")
using .Sparsity

include("Cluster/Pipeline.jl")
using .Pipeline

# SoftKATE's training helper pulls metrics + pipeline for the
# optional best-checkpoint tracking, so it has to sit after the
# Eval/ and Cluster/ includes.
include("Models/SoftKATE.jl")
using .SoftKATE

include("AutoTune.jl")
using .AutoTune

include("RCA.jl")
using .RCA

include("CLI.jl")
using .CLI

function __init__()
    PersistenceGlue.register_all!()
end

export KATE, DeepKATE, DenoisingAE, SoftKATE, Framing, Masking, Dedup,
       Featurise, Episodes, Baselines, Instance, Sequence, SeqLSTM,
       VQVAE, SimCSE, Metrics, Compression, Harness, CV, Sparsity,
       Pipeline, Drain3, Canonical, Persistence, PersistenceGlue,
       AutoTune, RCA, CLI, Purity, Typing, Merge, Rust

# ---------------------------------------------------------------------------
# Precompile workload
# ---------------------------------------------------------------------------
#
# `@compile_workload` runs at *precompile* time so the code paths it
# touches land in the package image and first-call latency in the
# interactive session drops close to zero. Everything expensive that
# users actually run on the hot path belongs here; anything that
# pulls in Python (UMAP, HDBSCAN) or the Rust crate (which may not be
# built yet) stays out.
using PrecompileTools: PrecompileTools

PrecompileTools.@setup_workload begin
    using Random: MersenneTwister
    using Lux: Lux, Chain, Dense, Embedding, Recurrence, LSTMCell
    using .Framing: parse_frame
    using .DeepKATE: deep_kate, deep_kate_loss, latent_layer
    using .DenoisingAE: denoising_ae, denoising_ae_loss
    using .SeqLSTM: seq_lstm, seq_lstm_loss, predict_next, PeepholeLSTM
    using .KATE: KCompetetive
    using .Episodes: mv_span, mt_span
    using .Instance: anomaly_score, reconstruction_error_abs
    using .Sparsity: sparsity_clusters
    using .Pipeline: l2_normalise, kmeans_cluster
    using .Metrics: parsing_accuracy, group_accuracy, grouping_f1,
                    template_group_f1, nmi, ari, purity, v_measure, to_labels
    using .Compression: dictionary_size, bpc_gzip, bpc_dictionary,
                        codebook_perplexity
    using .CV: time_ordered_split, time_ordered_kfold,
               per_host_split, stratified_by

    rng = MersenneTwister(0)

    # We skip Zygote-gradient warmup in the precompile workload: Zygote
    # generates a fresh pullback per call-site, so precompiling one
    # gradient invocation doesn't help subsequent ones. The TTFX win
    # comes from baking in the forward + loss paths and the Julia-side
    # clustering/metric code — Zygote overhead stays but only bites on
    # each new gradient call-site.

    PrecompileTools.@compile_workload begin
        # Framing — every source type gets parsed so the @inbounds
        # byte-dispatch paths precompile.
        parse_frame("<165>1 2024-01-01T00:00:00Z host app 1 - " *
                    "[sd@1 k=\"v\"] msg")
        parse_frame("2024-01-01T00:00:00.000Z stdout F hello")
        parse_frame("{\"log\":\"x\",\"stream\":\"stdout\",\"time\":\"t\"}")
        parse_frame("plain text line")

        # DeepKATE — build + forward in testmode (Dropout / KATE
        # competition bypassed so we don't trip the Lux "training=true
        # outside autodiff" warning during precompile). The loss path
        # itself is exercised in the test suite, not here. Cover both
        # the thesis-default topology and a widened-bottleneck variant
        # so TTFX stays low whichever path the user opts into.
        m = deep_kate(16; latent = 2, k1 = 4)
        ps, st = Lux.setup(rng, m)
        st_eval = Lux.testmode(st)
        x = rand(rng, Float32, 16, 2)
        m(x, ps, st_eval)

        m_wide = deep_kate(16; hidden = [32, 16], latent = 8, k1 = 8)
        ps_w, st_w = Lux.setup(rng, m_wide)
        m_wide(x, ps_w, Lux.testmode(st_w))

        # DenoisingAE — forward + loss in testmode.
        m_d = denoising_ae(16; hidden = 8, latent = 4)
        ps_d, st_d = Lux.setup(rng, m_d)
        denoising_ae_loss(m_d, ps_d, Lux.testmode(st_d), x;
                          mask_rate = 0.2, σ = 0.1, rng = rng)

        # SeqLSTM — plain and peephole variants, testmode only.
        m1 = seq_lstm(8; embed = 4, hidden = 6)
        ps1, st1 = Lux.setup(rng, m1)
        st1_eval = Lux.testmode(st1)
        seq = rand(rng, 1:8, 3, 2)
        seq_lstm_loss(m1, ps1, st1_eval, seq)
        predict_next(m1, ps1, st1, seq)

        m2 = seq_lstm(8; embed = 4, hidden = 6, peephole = true)
        ps2, st2 = Lux.setup(rng, m2)
        seq_lstm_loss(m2, ps2, Lux.testmode(st2), seq)

        # Episode miners on a tiny sequence.
        small = [1, 2, 3, 1, 2, 3]
        mv_span(small; min_sup = 2)
        mt_span(small, Dict(1 => 1.0, 2 => 1.0, 3 => 1.0);
                max_time_duration = 3)

        # Anomaly scoring.
        anomaly_score(m, ps, st, x;
                      weights = (abs = 1.0, sq = 1.0, latent = 0.0))

        # Sparsity + Pipeline (Julia-only clusterers).
        Z = rand(rng, Float32, 4, 10)
        sparsity_clusters(Z, 2)
        l2_normalise(Z)
        kmeans_cluster(Z, 2)

        # Metrics and compression on tiny string / int inputs.
        pred = ["a", "a", "b", "b"]
        gold = ["x", "x", "y", "y"]
        parsing_accuracy(pred, gold)
        group_accuracy(pred, gold)
        grouping_f1(pred, gold)
        template_group_f1(pred, gold)
        nmi(pred, gold); ari(pred, gold); purity(pred, gold); v_measure(pred, gold)
        to_labels(pred)
        dictionary_size(pred)
        bpc_gzip(["hello world" for _ in 1:4])
        bpc_dictionary(["hello world" for _ in 1:4], pred)
        codebook_perplexity(pred)

        # CV splits.
        time_ordered_split(10)
        time_ordered_kfold(10, 3)
        per_host_split(["a", "a", "b", "b", "c"]; held_out = "c")
        stratified_by(pred; rng = rng)

        # KATE layer on a small input.
        kl = KCompetetive(8, 4, tanh)
        psk, stk = Lux.setup(rng, kl)
        kl(rand(rng, Float32, 8), psk, Lux.testmode(stk))
    end
end

end # module LogClustering
