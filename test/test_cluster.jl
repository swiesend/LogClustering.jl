using Test
using Random
using LogClustering
using LogClustering.Sparsity
using LogClustering.Pipeline

const RNG_CLU = Random.MersenneTwister(7)

@testset "Cluster.Sparsity" begin
    @testset "top_k_indices — magnitude-ranked, sorted ascending" begin
        @test top_k_indices([0.1, 0.9, -0.5, 0.0, -0.9], 2) == (2, 5)
        @test top_k_indices([3.0, -1.0, 2.0, -4.0], 3) == (1, 3, 4)
    end

    @testset "top_k_indices — signed preserves sign" begin
        lab = top_k_indices([0.9, -0.5, -0.9], 2; signed = true)
        # Two largest |·| are indices 1 (+) and 3 (−)
        @test lab == ((1, Int8(1)), (3, Int8(-1)))
    end

    @testset "sparsity_labels + sparsity_clusters — stable first-seen ids" begin
        # Columns 1 and 3 share the same top-2 active set (indices 1,3);
        # column 2 has a distinct active set (indices 1,2).
        Z = Float32[
            1.0  1.0  1.0
            0.1  0.9  0.2
            0.8  0.05 0.9
            0.0  0.1  0.0
        ]
        labels = sparsity_labels(Z, 2)
        @test labels[1] == (1, 3)
        @test labels[2] == (1, 2)
        @test labels[3] == (1, 3)

        assign, label_of = sparsity_clusters(Z, 2)
        @test length(assign) == 3
        @test assign[1] == 1 && assign[3] == 1          # same cluster
        @test assign[2] == 2                             # different
        @test label_of[assign[1]] == (1, 3)
        @test label_of[assign[2]] == (1, 2)
    end

    @testset "sparsity_clusters — k = 1 collapses to argmax channel" begin
        Z = Float32[
            0.1 0.9 0.1
            0.9 0.1 0.9
        ]
        assign, _ = sparsity_clusters(Z, 1)
        @test assign == [1, 2, 1]
    end
end

@testset "Cluster.Pipeline" begin
    @testset "l2_normalise — every column is unit ℓ₂" begin
        X = Float32[1 2 0; 0 2 0; 0 1 0]
        Y = l2_normalise(X)
        @test all(isapprox(sum(abs2, Y[:, j]), 1.0; atol = 1e-6)
                  for j in 1:2)
        @test Y[:, 3] == zeros(Float32, 3)               # zero column stays zero
    end

    @testset "l2_normalise! — in-place" begin
        X = Float32[3 0; 4 0]
        l2_normalise!(X)
        @test X[1, 1] ≈ 3 / 5
        @test X[2, 1] ≈ 4 / 5
    end

    @testset "kmeans_cluster — recovers two well-separated blobs" begin
        blob1 = 0.05f0 .* randn(RNG_CLU, Float32, 4, 25) .+ Float32[2, 2, 0, 0]
        blob2 = 0.05f0 .* randn(RNG_CLU, Float32, 4, 25) .+ Float32[-2, -2, 0, 0]
        X = hcat(blob1, blob2)
        r = kmeans_cluster(X, 2)
        @test r.converged
        @test size(r.centers) == (4, 2)
        # Clusters should split the 50 samples 25/25 (labels may be swapped).
        counts = sort!([count(==(c), r.assignments) for c in unique(r.assignments)])
        @test counts == [25, 25]
    end

    @testset "kmeans_cluster — argument validation" begin
        @test_throws ArgumentError kmeans_cluster(randn(3, 4), 0)
        @test_throws ArgumentError kmeans_cluster(randn(3, 2), 5)   # k > n
    end
end
