# Utility functions for field tracing.

using Random
using Statistics: mean

export select_seeds, add_arrow

"""
	 select_seeds(x, y; nSeed=100)

Generate `nSeed` seeding points randomly in the grid range `x` and `y`.
If you specify `nSeed`, use the keyword input, otherwise it will be
overloaded by the 3D version seed generation function!
"""
function select_seeds(x, y; nSeed=100)
   xmin,xmax = extrema(x)
   ymin,ymax = extrema(y)

   xstart = rand(MersenneTwister(0),nSeed)*(xmax-xmin) .+ xmin
   ystart = rand(MersenneTwister(1),nSeed)*(ymax-ymin) .+ ymin
   seeds = zeros(eltype(x[1]),2,nSeed)
   for i in eachindex(xstart)
      seeds[1,i] = xstart[i]
      seeds[2,i] = ystart[i]
   end
   return seeds
end

function select_seeds(x, y, z; nSeed=100)
   xmin, xmax = extrema(x)
   ymin, ymax = extrema(y)
   zmin, zmax = extrema(z)

   xstart = rand(MersenneTwister(0), nSeed) * (xmax - xmin) .+ xmin
   ystart = rand(MersenneTwister(1), nSeed) * (ymax - ymin) .+ ymin
   zstart = rand(MersenneTwister(2), nSeed) * (zmax - zmin) .+ zmin
   seeds = zeros(eltype(x[1]), 3, nSeed)
   for i in eachindex(xstart)
      seeds[1,i] = xstart[i]
      seeds[2,i] = ystart[i]
      seeds[3,i] = zstart[i]
   end
   return seeds
end

"Add an arrow to a Line2D object `line` from Matplotlib."
function add_arrow(line, size=12)

   color = line.get_color()

   xdata = line.get_xdata()
   ydata = line.get_ydata()

   position = mean(xdata)
   # find closest index
   start_ind = argmin(abs.(xdata .- position))

   end_ind = start_ind + 1

   line.axes.annotate("",
      xytext=(xdata[start_ind], ydata[start_ind]),
      xy=(xdata[end_ind], ydata[end_ind]),
      arrowprops=Dict("arrowstyle"=>"-|>", "color"=>color),
      size=size
   )
end