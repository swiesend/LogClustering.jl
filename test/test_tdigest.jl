using Test
using LogClustering
using LogClustering.TDigests
using LogClustering.TDigests: TDigest, quantile, cdf, mean, var, std,
                               count, serialize, deserialize, merge!
using Random: MersenneTwister

# Exact quantile of a sorted vector, for accuracy comparison.
_exact_q(v, q) = (s = sort(v); s[clamp(ceil(Int, q * length(s)), 1, length(s))])

@testset "Stats.TDigest" begin

    @testset "empty digest is well-behaved" begin
        d = TDigest()
        @test count(d) == 0
        @test isnan(quantile(d, 0.5))
        @test isnan(mean(d))
        @test isnan(cdf(d, 0.0))
    end

    @testset "quantiles track the true distribution (uniform 1..10000)" begin
        d = TDigest()
        for i in 1:10_000
            push!(d, Float64(i))
        end
        @test count(d) == 10_000
        # Within 1% of the value range (100 units) on a uniform stream.
        for q in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
            @test abs(quantile(d, q) - q * 10_000) < 150
        end
        @test quantile(d, 0.0) <= 1.0 + 1e-6
        @test quantile(d, 1.0) >= 10_000 - 1e-6
    end

    @testset "mean/var/std are exact (independent of compression)" begin
        rng = MersenneTwister(1)
        xs = randn(rng, 5000) .* 3 .+ 7
        d = TDigest()
        for x in xs; push!(d, x); end
        @test mean(d) ≈ sum(xs) / length(xs) rtol=1e-9
        # population variance
        μ = sum(xs) / length(xs)
        popvar = sum((xs .- μ) .^ 2) / length(xs)
        @test var(d) ≈ popvar rtol=1e-9
        @test std(d) ≈ sqrt(popvar) rtol=1e-9
    end

    @testset "accuracy vs exact on a skewed (lognormal) stream" begin
        rng = MersenneTwister(2)
        xs = exp.(randn(rng, 20_000))     # heavy right tail — the alerting case
        d = TDigest()
        for x in xs; push!(d, x); end
        # p99 relative error under 5% on a heavy tail is the property that
        # matters for auto:p99 thresholds.
        p99_exact = _exact_q(xs, 0.99)
        @test abs(quantile(d, 0.99) - p99_exact) / p99_exact < 0.05
    end

    @testset "merge! equals a single digest over the concatenation" begin
        a = TDigest(); b = TDigest(); whole = TDigest()
        for i in 1:5000;      push!(a, Float64(i));     push!(whole, Float64(i)); end
        for i in 5001:10_000; push!(b, Float64(i));     push!(whole, Float64(i)); end
        merge!(a, b)
        @test count(a) == count(whole) == 10_000
        @test abs(quantile(a, 0.99) - quantile(whole, 0.99)) < 200
        @test mean(a) ≈ mean(whole) rtol=1e-9
    end

    @testset "serialize / deserialize round-trip" begin
        d = TDigest()
        for i in 1:3000; push!(d, sqrt(Float64(i))); end
        s = serialize(d)
        d2 = deserialize(s)
        @test count(d2) == count(d)
        @test mean(d2) ≈ mean(d) rtol=1e-9
        @test quantile(d2, 0.9) ≈ quantile(d, 0.9) rtol=1e-6
    end

    @testset "deserialize rejects malformed input" begin
        @test_throws ArgumentError deserialize("not;enough;fields")
    end

    @testset "memory stays bounded under a long stream" begin
        d = TDigest(; compression = 100.0)
        for i in 1:200_000; push!(d, Float64(i % 1000)); end
        # centroid count is bounded by ~compression, nowhere near 200k.
        @test length(d.centroids) <= 300
    end

    @testset "warm quantile is allocation-free (no sort-per-fire)" begin
        # The P1 point: auto:* thresholds used to `sort` the whole
        # reservoir on every rule fire (O(n) allocation). A t-digest
        # quantile is a scalar walk over the bounded centroid list — once
        # the buffer is flushed, it allocates nothing.
        d = TDigest(; compression = 100.0)
        reservoir = Float64[]
        for i in 1:20_000
            x = Float64(i % 1000)
            push!(d, x); Base.push!(reservoir, x)
        end
        quantile(d, 0.99)                       # warm: flush the buffer once
        a_digest = @allocated quantile(d, 0.99)
        a_sort   = @allocated sort(reservoir)   # the old per-fire cost
        @test a_digest < 512                    # scalar walk — no O(n) copy
        @test a_sort > 50_000                   # 20k Float64 sort copies ~160 KB
        @test a_digest < a_sort ÷ 50            # orders of magnitude cheaper
    end

end
