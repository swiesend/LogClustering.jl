using Test
using Random
using Statistics
using Lux
using Zygote
using LogClustering
using LogClustering.Transformer:
    Transformer, RMSNorm, GQAAttention, SwiGLUFFN, TransformerBlock,
    transformer_encoder, transformer_decoder,
    transformer_decoder_loss, transformer_encoder_loss,
    embed_sequences,
    rope_cache, apply_rope, rotate_half, d_ff_swiglu

# Qualify the names that clash with SeqLSTM / DeepKATE exports already
# in scope via `using LogClustering`.
const tx_predict_next = Transformer.predict_next
const tx_latent_layer = Transformer.latent_layer

const RNG_TX = Random.MersenneTwister(17)

# Reuse the recursive-SGD helper from test_seqlstm by re-defining locally;
# tests that include each other end up sharing the test-set tree.
function _sgd_step_tx(ps, grads, lr)
    if grads === nothing
        return ps
    elseif ps isa AbstractArray
        return ps .- lr .* grads
    elseif ps isa NamedTuple
        ks = keys(ps)
        return NamedTuple{ks}(map(k -> _sgd_step_tx(getfield(ps, k),
                                                    hasproperty(grads, k) ? getfield(grads, k) : nothing,
                                                    lr), ks))
    else
        return ps
    end
end

@testset "Transformer" begin

    @testset "RMSNorm matches manual reference" begin
        l = RMSNorm(8)
        ps, st = Lux.setup(RNG_TX, l)
        @test size(ps.g) == (8,)
        x = randn(RNG_TX, Float32, 8, 5, 3)
        y, _ = l(x, ps, st)
        @test size(y) == size(x)
        rms = sqrt.(mean(x .^ 2; dims = 1) .+ 1f-6)
        ref = (x ./ rms) .* ps.g
        @test isapprox(y, ref; atol = 1f-5)
        # Gradient flows through `g`.
        g = Zygote.gradient(p -> sum(abs2, first(l(x, p, st))), ps)[1]
        @test g.g !== nothing
        @test any(!iszero, g.g)
    end

    @testset "RoPE cache shape and rotate_half identities" begin
        cos_full, sin_full = rope_cache(8, 16)
        @test size(cos_full) == (8, 16)
        @test size(sin_full) == (8, 16)
        # Position 0 is identity rotation: cos == 1, sin == 0.
        @test all(cos_full[:, 1] .== 1)
        @test all(sin_full[:, 1] .== 0)
        # `rotate_half` swaps halves with a sign flip.
        x = collect(reshape(Float32.(1:8), 8, 1))
        rh = rotate_half(x)
        @test rh[1:4, 1] == -Float32.(5:8)
        @test rh[5:8, 1] ==  Float32.(1:4)
        # Rotation preserves Euclidean norm of each (head_dim, t)-column.
        q = randn(RNG_TX, Float32, 8, 16, 3)                    # (head_dim, T, batch)
        q_rot = apply_rope(q, cos_full, sin_full)
        norms_before = sqrt.(sum(q .^ 2; dims = 1))
        norms_after  = sqrt.(sum(q_rot .^ 2; dims = 1))
        @test isapprox(norms_before, norms_after; atol = 1f-4)
    end

    @testset "GQAAttention with n_kv_heads == n_heads behaves like MHA" begin
        l = GQAAttention(16; n_heads = 4, n_kv_heads = 4,
                         max_seq_len = 32, causal = false)
        ps, st = Lux.setup(RNG_TX, l)
        @test size(ps.wq) == (16, 16)
        @test size(ps.wk) == (16, 16)
        @test size(ps.wv) == (16, 16)
        @test size(ps.wo) == (16, 16)
        x = randn(RNG_TX, Float32, 16, 6, 2)
        y, _ = l(x, ps, st)
        @test size(y) == (16, 6, 2)
        @test all(isfinite, y)
        g = Zygote.gradient(p -> sum(abs2, first(l(x, p, st))), ps)[1]
        @test any(!iszero, g.wq)
        @test any(!iszero, g.wo)
    end

    @testset "GQAAttention with n_kv_heads < n_heads (group=4)" begin
        l = GQAAttention(16; n_heads = 4, n_kv_heads = 1,
                         max_seq_len = 32, causal = false)
        ps, st = Lux.setup(RNG_TX, l)
        @test size(ps.wk) == (4,  16)            # head_dim=4, 1 kv head
        @test size(ps.wv) == (4,  16)
        x = randn(RNG_TX, Float32, 16, 5, 2)
        y, _ = l(x, ps, st)
        @test size(y) == (16, 5, 2)
        g = Zygote.gradient(p -> sum(abs2, first(l(x, p, st))), ps)[1]
        @test any(!iszero, g.wk)
        @test any(!iszero, g.wv)
    end

    @testset "Causal mask: future tokens don't leak into past positions" begin
        m = transformer_decoder(12; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 32, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq_a = rand(RNG_TX, 1:12, 6, 2)
        seq_b = copy(seq_a)
        seq_b[end, :] .= 1                                       # perturb final token only
        h_a, _ = m(seq_a, ps, Lux.testmode(st))
        h_b, _ = m(seq_b, ps, Lux.testmode(st))
        # Earlier positions must be byte-identical (fp tolerance).
        diff = maximum(abs.(h_a[:, 1:(end - 1), :] .- h_b[:, 1:(end - 1), :]))
        @test diff < 1f-5
        # And the final position must differ.
        diff_last = maximum(abs.(h_a[:, end, :] .- h_b[:, end, :]))
        @test diff_last > 1f-5
    end

    @testset "Encoder is bidirectional (no causal mask)" begin
        m = transformer_encoder(12; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 4,
                                max_seq_len = 32, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq_a = rand(RNG_TX, 1:12, 6, 2)
        seq_b = copy(seq_a); seq_b[end, :] .= 1
        h_a, _ = m(seq_a, ps, Lux.testmode(st))
        h_b, _ = m(seq_b, ps, Lux.testmode(st))
        # Bidirectional: changing the last token DOES affect earlier positions.
        @test maximum(abs.(h_a[:, 1, :] .- h_b[:, 1, :])) > 1f-5
    end

    @testset "SwiGLUFFN shape and gradient" begin
        l = SwiGLUFFN(16; d_ff = 32)
        ps, st = Lux.setup(RNG_TX, l)
        @test size(ps.w1) == (32, 16)
        @test size(ps.w3) == (32, 16)
        @test size(ps.w2) == (16, 32)
        x = randn(RNG_TX, Float32, 16, 5, 2)
        y, _ = l(x, ps, st)
        @test size(y) == (16, 5, 2)
        g = Zygote.gradient(p -> sum(abs2, first(l(x, p, st))), ps)[1]
        @test any(!iszero, g.w1)
        @test any(!iszero, g.w2)
        @test any(!iszero, g.w3)
        # `d_ff_swiglu` rounds to a multiple of 64.
        @test d_ff_swiglu(128; ffn_mult = 4) % 64 == 0
        @test d_ff_swiglu(64) >= 64
    end

    @testset "Decoder loss is non-negative and differentiable" begin
        m = transformer_decoder(8; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:8, 6, 3)
        loss, _ = transformer_decoder_loss(m, ps, st, seq)
        @test loss isa Real
        @test loss >= 0
        @test isfinite(loss)
        @test_throws ArgumentError transformer_decoder_loss(m, ps, st,
            rand(RNG_TX, 1:8, 1, 2))
        g = Zygote.gradient(p -> first(transformer_decoder_loss(m, p, st, seq)), ps)[1]
        @test g !== nothing
        @test any(!iszero, g.layer_1.weight)                    # tied LM head
    end

    @testset "Encoder MLM loss is non-negative and differentiable" begin
        m = transformer_encoder(8; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 4,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:8, 6, 3)
        loss, _ = transformer_encoder_loss(m, ps, st, seq;
                                           mask_rate = 0.3, rng = copy(RNG_TX))
        @test loss isa Real
        @test loss >= 0
        @test isfinite(loss)
        g = Zygote.gradient(p ->
                first(transformer_encoder_loss(m, p, st, seq;
                                               mask_rate = 0.3,
                                               rng = MersenneTwister(99))), ps)[1]
        @test g !== nothing
        @test any(!iszero, g.layer_1.weight)
    end

    @testset "Decoder memorises a tiny sequence" begin
        # 200 SGD steps on a 4-token toy sequence should drop NLL well
        # below the uniform baseline log(vocab=2).
        m = transformer_decoder(2; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 8, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = reshape(Int[1, 2, 1, 2, 1], :, 1)
        lr = Float32(0.05)
        final_loss = Inf32
        for _ in 1:200
            (loss, _), back = Zygote.pullback(p ->
                transformer_decoder_loss(m, p, st, seq), ps)
            g = back((one(loss), nothing))[1]
            ps = _sgd_step_tx(ps, g, lr)
            final_loss = loss
        end
        @test final_loss < log(2)
    end

    @testset "embed_sequences shape and determinism" begin
        m = transformer_encoder(10; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:10, 6, 4)
        z1 = embed_sequences(m, ps, st, seq)
        z2 = embed_sequences(m, ps, st, seq)
        @test size(z1) == (16, 4)
        @test z1 == z2                                          # testmode → no dropout noise
        @test size(embed_sequences(m, ps, st, seq; pool = :last)) == (16, 4)
        @test_throws ArgumentError embed_sequences(m, ps, st, seq; pool = :foo)
    end

    @testset "predict_next returns valid token ids" begin
        m = transformer_decoder(11; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:11, 6, 5)
        ids = tx_predict_next(m, ps, st, seq)
        @test length(ids) == 5
        @test all(1 .<= ids .<= 11)
    end

    @testset "latent_layer points at the final RMSNorm" begin
        m = transformer_encoder(8; d_model = 16, n_layers = 3,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        # Embedding + 3 blocks + final RMSNorm = 5 layers.
        @test tx_latent_layer(m) == 5
        m2 = transformer_decoder(8; d_model = 16, n_layers = 4,
                                 n_heads = 4, n_kv_heads = 2,
                                 max_seq_len = 16, dropout = 0.0)
        @test tx_latent_layer(m2) == 6
    end

    @testset "Constructor argument validation" begin
        @test_throws ArgumentError GQAAttention(15; n_heads = 4)        # not divisible
        @test_throws ArgumentError GQAAttention(16; n_heads = 4, n_kv_heads = 3)
        @test_throws ArgumentError GQAAttention(12; n_heads = 4)        # head_dim=3 odd
    end

    @testset "Persistence round-trip — encoder" begin
        m = transformer_encoder(10; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:10, 6, 3)
        h_before, _ = m(seq, ps, Lux.testmode(st))
        path = tempname() * ".jld2"
        try
            LogClustering.PersistenceGlue.save_transformer_encoder(
                path, m, ps, st;
                vocab_size = 10, d_model = 16, n_layers = 2,
                n_heads = 4, n_kv_heads = 2,
                ffn_mult = 4, max_seq_len = 16, dropout = 0.0,
                seqlen = 6,
                metadata = Dict("source" => "test"))
            bundle = LogClustering.Persistence.load(path)
            @test bundle.kind === :transformer_encoder
            art = LogClustering.Persistence.rehydrate(bundle)
            h_after, _ = art.model(seq, art.ps, Lux.testmode(art.st))
            @test isapprox(h_before, h_after; rtol = 1f-6)
            @test art.seqlen == 6
        finally
            isfile(path) && rm(path; force = true)
        end
    end

    @testset "Persistence round-trip — decoder" begin
        m = transformer_decoder(10; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        seq = rand(RNG_TX, 1:10, 6, 3)
        loss_before, _ = transformer_decoder_loss(m, ps, st, seq)
        path = tempname() * ".jld2"
        try
            LogClustering.PersistenceGlue.save_transformer_decoder(
                path, m, ps, st;
                vocab_size = 10, d_model = 16, n_layers = 2,
                n_heads = 4, n_kv_heads = 2,
                ffn_mult = 4, max_seq_len = 16, dropout = 0.0,
                seqlen = 6)
            bundle = LogClustering.Persistence.load(path)
            @test bundle.kind === :transformer_decoder
            art = LogClustering.Persistence.rehydrate(bundle)
            loss_after, _ = transformer_decoder_loss(art.model, art.ps, art.st, seq)
            @test isapprox(loss_before, loss_after; rtol = 1f-6)
        finally
            isfile(path) && rm(path; force = true)
        end
    end

    @testset "Persistence: dispatched save(path, model, ps, st; kind=:…)" begin
        m = transformer_decoder(8; d_model = 16, n_layers = 2,
                                n_heads = 4, n_kv_heads = 2,
                                max_seq_len = 16, dropout = 0.0)
        ps, st = Lux.setup(RNG_TX, m)
        path = tempname() * ".jld2"
        try
            # Goes through the type-dispatched `save(path, model, ps, st; kind=…)`
            LogClustering.PersistenceGlue.save(path, m, ps, st;
                kind = :transformer_decoder,
                vocab_size = 8, d_model = 16, n_layers = 2,
                n_heads = 4, n_kv_heads = 2,
                ffn_mult = 4, max_seq_len = 16, dropout = 0.0,
                seqlen = 4)
            art = LogClustering.Persistence.load_and_rehydrate(path)
            @test art.model isa Lux.Chain
            @test art.seqlen == 4
        finally
            isfile(path) && rm(path; force = true)
        end
    end
end
