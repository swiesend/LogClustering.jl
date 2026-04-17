using Test
using LinearAlgebra: I
using Random
using Lux
using Zygote
using LogClustering
using LogClustering.VQVAE: VectorQuantizer, vq_vae, vq_vae_loss, assign_codes

const RNG_VQ = Random.MersenneTwister(23)

@testset "VQVAE" begin
    @testset "VectorQuantizer — construction + parameter shape" begin
        l = VectorQuantizer(8, 16)
        ps, st = Lux.setup(RNG_VQ, l)
        @test size(ps.codebook) == (8, 16)
        @test haskey(st, :codes)
        @test_throws ArgumentError VectorQuantizer(0, 4)
        @test_throws ArgumentError VectorQuantizer(4, 0)
    end

    @testset "VectorQuantizer — forward snaps to a codebook column" begin
        l = VectorQuantizer(4, 6)
        ps, st = Lux.setup(RNG_VQ, l)
        X = randn(RNG_VQ, Float32, 4, 5)
        Y, st_new = l(X, ps, st)
        @test size(Y) == size(X)
        @test length(st_new.codes) == 5
        @test all(1 .<= st_new.codes .<= 6)
        # Y columns must equal the chosen codebook vectors
        # (up to the straight-through additive x − sg(x) identity).
        @test all(Y[:, j] ≈ ps.codebook[:, st_new.codes[j]] for j in 1:5)
    end

    @testset "VectorQuantizer — dimension mismatch raises" begin
        l = VectorQuantizer(4, 6)
        ps, st = Lux.setup(RNG_VQ, l)
        @test_throws DimensionMismatch l(randn(Float32, 3, 2), ps, st)
    end

    @testset "vq_vae — forward shape preserved, assign_codes in range" begin
        m = vq_vae(8; codebook_size = 10, embed_dim = 4, hidden = 16)
        ps, st = Lux.setup(RNG_VQ, m)
        X = rand(RNG_VQ, Float32, 8, 12)
        Y, _ = m(X, ps, st)
        @test size(Y) == (8, 12)
        codes = assign_codes(m, ps, st, X)
        @test length(codes) == 12
        @test all(1 .<= codes .<= 10)
    end

    @testset "vq_vae_loss is differentiable (Zygote reaches encoder + codebook)" begin
        m = vq_vae(6; codebook_size = 8, embed_dim = 3, hidden = 10)
        ps, st = Lux.setup(RNG_VQ, m)
        X = rand(RNG_VQ, Float32, 6, 4)
        loss, _ = vq_vae_loss(m, ps, st, X)
        @test isfinite(loss)
        @test loss >= 0
        g = Zygote.gradient(p -> first(vq_vae_loss(m, p, st, X)), ps)[1]
        @test g !== nothing
        # Encoder Dense (layer_1) must receive a non-zero gradient —
        # this is the signal that the straight-through estimator is
        # wired correctly.
        @test any(!iszero, g.layer_1.weight)
        # Codebook gets its gradient through the codebook-loss path.
        @test any(!iszero, g.layer_3.codebook)
    end

    @testset "VectorQuantizer — identity codebook recovers ground-truth codes" begin
        # Most meaningful cluster-recovery test we can run without
        # actually training: hand a `VectorQuantizer` an identity-shaped
        # codebook, feed each column of an identity-shaped input, and
        # check the assignments are the diagonal. An *un*-trained
        # full `vq_vae` Chain collapses codes in a way that depends on
        # encoder randomness — a training loop would fix that but
        # would turn a fast unit test into a slow one.
        l = VectorQuantizer(4, 4)
        ps = (codebook = Matrix{Float32}(I, 4, 4),)    # e_k = one-hot at k
        _, st0 = Lux.setup(Random.MersenneTwister(0), l)
        X = Matrix{Float32}(I, 4, 4)                    # x_j = one-hot at j
        _, st_new = l(X, ps, st0)
        @test st_new.codes == [1, 2, 3, 4]
        # Shuffled input columns → permuted codes.
        perm = [3, 1, 4, 2]
        _, st_perm = l(X[:, perm], ps, st0)
        @test st_perm.codes == perm
    end
end
