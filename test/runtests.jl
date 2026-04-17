using Test
using LogClustering

@testset "LogClustering.jl" begin
    include("test_KATE.jl")
    include("test_deepkate.jl")
    include("test_framing.jl")
    include("test_episodes.jl")
    include("test_rust.jl")
end
