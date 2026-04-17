using Test
using Random
using Lux
using LogClustering
using LogClustering.Persistence
using LogClustering.PersistenceGlue
using LogClustering.KATE: KCompetetive
using LogClustering.DeepKATE: deep_kate
using LogClustering.SeqLSTM: seq_lstm, seq_lstm_loss
using LogClustering.VQVAE: vq_vae, assign_codes
using LogClustering.Drain3: Drain, process!, parse_all
using LogClustering.Instance: ValueNoveltyDetector, update!, value_novelty
using LogClustering.Dedup: DedupState, is_new!, contains
using LogClustering.Masking: SlotValue

const RNG_PERS = Random.MersenneTwister(7)

"Round-trip `(model, ps, st)` forward values to within Float32 precision."
function _forward_agrees(m_before, ps_before, st_before,
                         m_after,  ps_after,  st_after, X)
    y_before, _ = m_before(X, ps_before,  Lux.testmode(st_before))
    y_after,  _ = m_after(X,  ps_after,   Lux.testmode(st_after))
    return size(y_before) == size(y_after) && all(isapprox.(y_before, y_after; rtol = 1e-5))
end

@testset "Persistence" begin
    @testset "schema version + metadata survive the trip" begin
        l = KCompetetive(8, 4, tanh; k = 4)
        ps, st = Lux.setup(RNG_PERS, l)
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, l, ps, st;
                                 metadata = Dict("note" => "smoke"))
            b = Persistence.load(path)
            @test b.kind == :kate
            @test b.schema_version == Persistence.SCHEMA_VERSION
            @test b.metadata["note"] == "smoke"
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "KCompetetive — round-trip forward" begin
        l = KCompetetive(10, 6, tanh; k = 6)
        ps, st = Lux.setup(RNG_PERS, l)
        X = randn(RNG_PERS, Float32, 10, 3)
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, l, ps, st)
            out = Persistence.load_and_rehydrate(path)
            @test _forward_agrees(l, ps, st, out.layer, out.ps, out.st, X)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "DeepKATE — round-trip forward and loss" begin
        m = deep_kate(12; latent = 2, k1 = 4, p = 0.4f0)
        ps, st = Lux.setup(RNG_PERS, m)
        X = rand(RNG_PERS, Float32, 12, 4)
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, m, ps, st;
                                 kind = :deep_kate,
                                 n = 12, latent = 2, k1 = 4, p = 0.4f0)
            out = Persistence.load_and_rehydrate(path)
            @test _forward_agrees(m, ps, st, out.model, out.ps, out.st, X)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "SeqLSTM — round-trip plain + peephole" begin
        for (peep, bidir) in ((false, false), (true, false))
            m = seq_lstm(8; embed = 4, hidden = 6,
                         peephole = peep, bidirectional = bidir)
            ps, st = Lux.setup(RNG_PERS, m)
            seq = rand(RNG_PERS, 1:8, 4, 2)
            y_before, _ = m(seq, ps, Lux.testmode(st))
            path = tempname() * ".jld2"
            try
                PersistenceGlue.save(path, m, ps, st;
                                     kind = :seq_lstm,
                                     vocab_size = 8, embed = 4, hidden = 6,
                                     peephole = peep, bidirectional = bidir)
                out = Persistence.load_and_rehydrate(path)
                y_after, _ = out.model(seq, out.ps, Lux.testmode(out.st))
                @test size(y_before) == size(y_after)
                @test all(isapprox.(y_before, y_after; rtol = 1e-5))
            finally
                isfile(path) && rm(path)
            end
        end
    end

    @testset "VQ-VAE — assign_codes is deterministic across save/load" begin
        m = vq_vae(8; codebook_size = 6, embed_dim = 3, hidden = 12)
        ps, st = Lux.setup(RNG_PERS, m)
        X = rand(RNG_PERS, Float32, 8, 10)
        codes_before = assign_codes(m, ps, st, X)
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, m, ps, st;
                                 kind = :vq_vae,
                                 n = 8, codebook_size = 6,
                                 embed_dim = 3, hidden = 12)
            out = Persistence.load_and_rehydrate(path)
            codes_after = assign_codes(out.model, out.ps, out.st, X)
            @test codes_before == codes_after
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "Drain — tree + clusters round-trip; parametrize restored via registry" begin
        d = Drain(; depth = 4, sim_th = 0.4)
        lines = [
            "INFO started pid 1001",
            "INFO started pid 1002",
            "ERROR connection refused from 10.0.0.1",
            "ERROR connection refused from 10.0.0.2",
        ]
        templates_before = parse_all(d, lines)
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, d)
            out = Persistence.load_and_rehydrate(path)
            @test out isa Drain
            @test out.depth == d.depth
            @test out.sim_th == d.sim_th
            @test length(out.clusters) == length(d.clusters)
            @test out.next_id == d.next_id
            # Running `process!` on the loaded parser must reproduce
            # the same cluster assignments as the original on the
            # same extra lines.
            extra = [
                "INFO started pid 5000",
                "DEBUG heartbeat",
            ]
            d_copy = deepcopy(d)  # fresh call to compare vs loaded
            ids_copy = [first(process!(d_copy, l)) for l in extra]
            ids_load = [first(process!(out, l))    for l in extra]
            @test ids_copy == ids_load
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "Drain — unknown parametrize raises on load" begin
        d = Drain(; parametrize = _ -> false)   # unregistered lambda
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, d)
            # The spec writes `:custom`; load errors because
            # :custom isn't in the callable registry.
            @test_throws ErrorException Persistence.load_and_rehydrate(path)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "ValueNoveltyDetector — state survives round-trip" begin
        det = ValueNoveltyDetector()
        update!(det, [SlotValue("USER", "alice", 1, 5)])
        update!(det, [SlotValue("NUM", "42", 1, 2)])
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, det)
            loaded = Persistence.load_and_rehydrate(path)
            @test loaded isa ValueNoveltyDetector
            @test loaded.seen == det.seen
            @test loaded.counts == det.counts
            # A fresh probe on the loaded detector agrees with the
            # original.
            probe = [SlotValue("USER", "attacker", 1, 8)]
            @test value_novelty(loaded, probe) == value_novelty(det, probe)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "DedupState — bits + counters survive round-trip" begin
        s = DedupState(; expected_n = 100, fpr = 1e-3)
        for w in ("a", "b", "c")
            is_new!(s, w)
        end
        path = tempname() * ".jld2"
        try
            PersistenceGlue.save(path, s)
            loaded = Persistence.load_and_rehydrate(path)
            @test loaded isa DedupState
            @test loaded.m == s.m
            @test loaded.k == s.k
            @test loaded.bits == s.bits
            @test loaded.observed == s.observed
            # Queries agree.
            @test contains(loaded, "a") == contains(s, "a")
            @test contains(loaded, "nonexistent") == contains(s, "nonexistent")
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "unknown kind errors with a helpful message" begin
        path = tempname() * ".jld2"
        try
            bundle = Persistence.PersistedBundle(
                :made_up, v"0.0.0", Persistence.SCHEMA_VERSION,
                NamedTuple(), nothing, Dict{String, Any}())
            Persistence._save_bundle(path, bundle)
            @test_throws ErrorException Persistence.load_and_rehydrate(path)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "newer schema errors on load" begin
        path = tempname() * ".jld2"
        try
            # Fake a future schema.
            bundle = Persistence.PersistedBundle(
                :drain, v"0.0.0", UInt8(Persistence.SCHEMA_VERSION + 1),
                NamedTuple(), nothing, Dict{String, Any}())
            Persistence._save_bundle(path, bundle)
            @test_throws ErrorException Persistence.load(path)
        finally
            isfile(path) && rm(path)
        end
    end
end
