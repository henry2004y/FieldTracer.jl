module FieldTracer

# Hongyang Zhou, hyzhou@umich.edu

using Meshes, Requires

include("structured2d.jl")
include("structured3d.jl")
include("unstructured2d.jl")
include("utility/seed.jl")

function __init__()
   @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" begin
      include("utility/pyplot.jl")
  end
end

end
