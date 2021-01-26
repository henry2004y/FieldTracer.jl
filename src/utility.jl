# Utility functions for field tracing.

export select_seeds

"""
	 select_seeds(x, y, nSeed=100)

Generate `nSeed` seeding points randomly in the grid range. If you specify
`nSeed`, use the keyword input, otherwise it will be overloaded by the 3D
version seed generation.
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