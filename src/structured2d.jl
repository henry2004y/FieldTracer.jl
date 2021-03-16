# 2D Field tracing on a regular grid.

export trace2d, trace2d_rk4, trace2d_euler

"""
    bilin_reg(x, y, Q00, Q01, Q10, Q11)

Bilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1)
Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.
"""
function bilin_reg(x, y, Q00, Q10, Q01, Q11)
   fout =
      Q00*(1.0-x)*(1.0-y) +
      Q10* x *    (1.0-y) +
      Q01* y *    (1.0-x) +
      Q11* x * y
end

"""
    grid_interp(x, y, field, ix, iy)

Interpolate a value at (x,y) in a field. `ix` and `iy` are indexes for x,y
locations (0-based).
"""
grid_interp(x, y, field::Array, ix, iy) =
   bilin_reg(x-ix, y-iy,
   field[ix+1, iy+1],
   field[ix+2, iy+1],
   field[ix+1, iy+2],
   field[ix+2, iy+2])

"""
    DoBreak(iloc, jloc, iSize, jSize)

Check to see if we should break out of an integration.
"""
function DoBreak(iloc, jloc, iSize, jSize)
   ibreak = false
   if iloc ≥ iSize-1 || jloc ≥ jSize-1; ibreak = true end
   if iloc < 0 || jloc < 0; ibreak = true end
   return ibreak
end

"Create unit vectors of field."
function normalize_field(ux, uy, dx, dy)
   fx, fy = similar(ux), similar(uy)
   dxInv, dyInv = 1/dx, 1/dy
   @inbounds for i = 1:length(ux)
      uInv = 1.0 / hypot(ux[i]*dxInv, uy[i]*dyInv)
      fx[i] = ux[i] * dxInv * uInv
      fy[i] = uy[i] * dyInv * uInv
   end
   return fx, fy
end

"""
    Euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast 2D tracing using Euler's method. It takes at most `maxstep` with step size
`ds` tracing the vector field given by `ux,uy` starting from `(startx,starty)`
in the Cartesian grid specified by ranges `xGrid` and `yGrid`.
Return footprints' coordinates in (`x`,`y`).
"""
function Euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

   @assert size(ux) == size(uy) "field array sizes must be equal!"

   x = Vector{eltype(startx)}(undef, maxstep)
   y = Vector{eltype(starty)}(undef, maxstep)

   iSize, jSize = size(xGrid,1), size(yGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   x[1] = (startx-xGrid[1]) / dx
   y[1] = (starty-yGrid[1]) / dy

   # Create unit vectors from full vector field
   f1, f2 = normalize_field(ux, uy, dx, dy)

   nstep = 0
   # Perform tracing using Euler's method
   for n = 1:maxstep-1
      # Find surrounding points
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])

      # Break if we leave the domain
      if DoBreak(ix, iy, iSize, jSize)
         nstep = n; break
      end

      # Interpolate unit vectors to current location
      fx = grid_interp(x[n], y[n], f1, ix, iy)
      fy = grid_interp(x[n], y[n], f2, ix, iy)

      if isnan(fx) || isnan(fy) || isinf(fx) || isinf(fy)
         nstep = n
         break
      end

      x[n+1] = x[n] + ds * fx
      y[n+1] = y[n] + ds * fy

      nstep = n
   end

   # Return traced points to original coordinate system.
   for i = 1:nstep
      x[i] = x[i]*dx + xGrid[1]
      y[i] = y[i]*dy + yGrid[1]
   end
   return x[1:nstep], y[1:nstep]
end

"""
    RK4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and
constant step size `ds`. See also [`Euler`](@ref).
"""
function RK4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

   @assert size(ux) == size(uy) "field array sizes must be equal!"

   x = Vector{eltype(startx)}(undef, maxstep)
   y = Vector{eltype(starty)}(undef, maxstep)

   iSize, jSize = size(xGrid,1), size(yGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   x[1] = (startx-xGrid[1]) / dx
   y[1] = (starty-yGrid[1]) / dy

   # Create unit vectors from full vector field
   fx, fy = normalize_field(ux, uy, dx, dy)

   nstep = 0
   # Perform tracing using RK4
   for n = 1:maxstep-1
      # See Euler's method for more descriptive comments.
      # SUBSTEP #1
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f1x = grid_interp(x[n], y[n], fx, ix, iy)
      f1y = grid_interp(x[n], y[n], fy, ix, iy)
      if isnan(f1x) || isnan(f1y) || isinf(f1x) || isinf(f1y)
         nstep = n; break
      end
      # SUBSTEP #2
      xpos = x[n] + f1x*ds/2.0
      ypos = y[n] + f1y*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f2x = grid_interp(xpos, ypos, fx, ix, iy)
      f2y = grid_interp(xpos, ypos, fy, ix, iy)

      if isnan(f2x) || isnan(f2y) || isinf(f2x) || isinf(f2y)
         nstep = n; break
      end
      # SUBSTEP #3
      xpos = x[n] + f2x*ds/2.0
      ypos = y[n] + f2y*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f3x = grid_interp(xpos, ypos, fx, ix, iy)
      f3y = grid_interp(xpos, ypos, fy, ix, iy)
      if isnan(f3x) || isnan(f3y) || isinf(f3x) || isinf(f3y)
         nstep = n; break
      end

      # SUBSTEP #4
      xpos = x[n] + f3x*ds
      ypos = y[n] + f3y*ds
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f4x = grid_interp(xpos, ypos, fx, ix, iy)
      f4y = grid_interp(xpos, ypos, fy, ix, iy)
      if isnan(f4x) || isnan(f4y) || isinf(f4x) || isinf(f4y)
         nstep = n; break
      end

      # Peform the full step using all substeps
      x[n+1] = x[n] + ds/6.0 * (f1x + f2x*2.0 + f3x*2.0 + f4x)
      y[n+1] = y[n] + ds/6.0 * (f1y + f2y*2.0 + f3y*2.0 + f4y)

      nstep = n
   end

   # Return traced points to original coordinate system.
   for i = 1:nstep
      x[i] = x[i]*dx + xGrid[1]
      y[i] = y[i]*dy + yGrid[1]
   end
   return x[1:nstep], y[1:nstep]
end

"""
	 trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy;
		 maxstep=20000, ds=0.01, gridType="ndgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of
the vector field. The field is integrated using Runge Kutta 4. Slower than
Euler, but more accurate. The higher accuracy allows for larger step sizes `ds`.
See also [`trace2d_euler`](@ref).
"""
function trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy;
   maxstep=20000, ds=0.01, gridType="ndgrid", direction="both")

   if gridType == "ndgrid"
      fx = fieldx
      fy = fieldy
   else # meshgrid
      fx = permutedims(fieldx)
      fy = permutedims(fieldy)
   end

   if direction == "forward"
      xt, yt = RK4(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
   elseif direction == "backward"
      xt, yt = RK4(maxstep, ds, startx, starty, gridx, gridy, -fx, -fy)
   else
      x1, y1 = RK4(floor(Int,maxstep/2), ds, startx, starty, gridx, gridy,
         -fx, -fy)
      blen = length(x1)
      x2, y2 = RK4(maxstep-blen, ds, startx, starty, gridx, gridy, fx, fy)
      # concatenate with duplicates removed
      xt = vcat(reverse!(x1), x2[2:end])
      yt = vcat(reverse!(y1), y2[2:end])
   end

   return xt, yt
end

"""
	 trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy;
		 maxstep=20000, ds=0.01, gridType="ndgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of
the vector field. The field is integrated using Euler's method, which is faster
but less accurate than RK4. Only valid for regular grid with coordinates' range
`gridx` and `gridy`. The field can be in both `meshgrid` or `ndgrid` (default)
format. Supporting `direction` of {"both","forward","backward"}.
"""
function trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy;
   maxstep=20000, ds=0.01, gridType="ndgrid", direction="both")

   if gridType == "ndgrid"
      fx = fieldx
      fy = fieldy
   else # meshgrid
      fx = permutedims(fieldx)
      fy = permutedims(fieldy)
   end

   if direction == "forward"
      xt, yt = Euler(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
   elseif direction == "backward"
      xt, yt = Euler(maxstep, ds, startx, starty, gridx, gridy, -fx, -fy)
   else
      x1, y1 = Euler(floor(Int,maxstep/2), ds, startx, starty, gridx, gridy,
         -fx, -fy)
      blen = length(x1)
      x2, y2 = Euler(maxstep-blen, ds, startx, starty, gridx, gridy, fx, fy)
      # concatenate with duplicates removed
      xt = vcat(reverse!(x1), x2[2:end])
      yt = vcat(reverse!(y1), y2[2:end])
   end

   return xt, yt
end

"""
    trace2d_euler(fieldx, fieldy, startx, starty, grid::CartesianGrid;
       kwargs...)
"""
function trace2d_euler(fieldx, fieldy, startx, starty, grid::CartesianGrid;
   kwargs...)

   gridmin = coordinates(minimum(grid))
   gridmax = coordinates(maximum(grid))
   Δx = spacing(grid)

   gridx = range(gridmin[1], gridmax[1], step=Δx[1])
   gridy = range(gridmin[2], gridmax[2], step=Δx[2])

   trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)
end

"""
	 trace2d(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)

2D stream tracing on structured mesh with field in 2D array and grid in range.
"""
trace2d(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...) =
   trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)