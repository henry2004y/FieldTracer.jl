module FieldTracer

using Meshes, Requires, MuladdMacro

export trace
export Euler, RK4

abstract type Algorithm end

struct Euler <: Algorithm end
struct RK4 <: Algorithm end

include("structured2d.jl")
include("structured3d.jl")
include("unstructured2d.jl")
include("utility/seed.jl")

"""
	 trace(fieldx, fieldy, startx, starty, gridx, gridy; alg=RK4(), kwargs...)

Stream tracing on structured mesh with field in 2D array and grid in range. The keyword
arguments are the same as in [`trace2d_euler`](@ref) and [`trace2d_rk4`](@ref).
"""

"""
	 trace(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
       alg=RK4(), kwargs...)

    trace(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;
		 alg=RK4(), maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

Stream tracing on structured mesh with field in 3D array and grid in range.
"""
function trace(args...; alg::Algorithm=RK4(), kwargs...)
   if length(args) ≤ 6 # 2D
      if alg isa RK4
         trace2d_rk4(args...; kwargs...)
      elseif alg isa Euler
         trace2d_euler(args...; kwargs...)
      end
   else # 3D 
      if alg isa RK4
         trace3d_rk4(args...; kwargs...)
      elseif alg isa Euler
         trace3d_euler(args...; kwargs...)
      end
   end

end

function __init__()
   @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" begin
      include("utility/pyplot.jl")
   end
end

end
