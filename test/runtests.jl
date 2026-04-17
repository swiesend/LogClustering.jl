using Test
using LogClustering

@testset "LogClustering.jl" begin
    include("test_KATE.jl")
    include("test_deepkate.jl")
    include("test_anomaly.jl")
    include("test_value_novelty.jl")
    include("test_seqlstm.jl")
    include("test_framing.jl")
    include("test_preproc.jl")
    include("test_episodes.jl")
    include("test_eval.jl")
    include("test_compression.jl")
    include("test_cv.jl")
    include("test_cluster.jl")
    include("test_drain.jl")
    include("test_canonical.jl")
    include("test_ad.jl")
    include("test_rust.jl")
end
