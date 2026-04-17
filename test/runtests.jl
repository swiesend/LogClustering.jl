using Test
using LogClustering

@testset "LogClustering.jl" begin
    include("test_KATE.jl")
    include("test_framing.jl")
end
