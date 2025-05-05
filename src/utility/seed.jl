# Utility functions for field tracing.

export select_seeds

"""
	 select_seeds(x, y; nsegment=(5, 5))

Generate uniform seeding points in the grid range `x` and `y` in `nsegment`. If `nsegment`
specified, use the keyword input, otherwise it will be overloaded by the 3D version seed
generation function!
"""
function select_seeds(x, y; nsegment = (5, 5))
	xmin, xmax = extrema(x)
	ymin, ymax = extrema(y)

	dx = (xmax - xmin) / nsegment[1]
	dy = (ymax - ymin) / nsegment[2]

	xrange = xmin .+ dx*((1:nsegment[1]) .- 0.5)
	yrange = ymin .+ dy*((1:nsegment[2]) .- 0.5)

	seeds = zeros(eltype(x[1]), 2, prod(nsegment))
	for i in 0:(prod(nsegment)-1)
		seeds[1, i+1] = xrange[i%nsegment[1]+1]
		seeds[2, i+1] = yrange[i÷nsegment[2]+1]
	end
	seeds
end

"""
	 select_seeds(x, y, z; nsegment=(5, 5, 5))

Generate uniform seeding points in the grid range `x`, `y` and `z` in `nsegment`.
"""
function select_seeds(x, y, z; nsegment = (5, 5, 5))
	xmin, xmax = extrema(x)
	ymin, ymax = extrema(y)
	zmin, zmax = extrema(z)

	dx = (xmax - xmin) / nsegment[1]
	dy = (ymax - ymin) / nsegment[2]
	dz = (zmax - zmin) / nsegment[3]

	xrange = xmin .+ dx*((1:nsegment[1]) .- 0.5)
	yrange = ymin .+ dy*((1:nsegment[2]) .- 0.5)
	zrange = zmin .+ dz*((1:nsegment[3]) .- 0.5)

	seeds = zeros(eltype(x[1]), 3, prod(nsegment))
	@inbounds for k in 1:nsegment[3], j in 1:nsegment[2], i in 1:nsegment[1]
		seeds[1, i+(j-1)*nsegment[2]+(k-1)*nsegment[2]*nsegment[3]] = xrange[i]
		seeds[2, i+(j-1)*nsegment[2]+(k-1)*nsegment[2]*nsegment[3]] = yrange[j]
		seeds[3, i+(j-1)*nsegment[2]+(k-1)*nsegment[2]*nsegment[3]] = zrange[k]
	end

	seeds
end
