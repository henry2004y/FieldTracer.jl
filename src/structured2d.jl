# 2D Field tracing on a regular grid.

using Random

export trace2d, trace2d_rk4, trace2d_eul

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
    DoBreak(iloc, jloc, iSize, jSize)

Check to see if we should break out of an integration.
"""
function DoBreak(iloc::T, jloc::T, iSize::T, jSize::T) where {T<:Integer}
   ibreak = false
   if iloc ≥ iSize-1 || jloc ≥ jSize-1; ibreak = true end
   if iloc < 0 || jloc < 0; ibreak = true end
   return ibreak
end

function DoBreak(iloc::T, jloc::T, kloc::T, iSize::T, jSize::T, kSize::T) where
   {T<:Integer}
   ibreak = false
   if iloc ≥ iSize-1 || jloc ≥ jSize-1 || kloc ≥ kSize-1; ibreak = true end
   if iloc < 0 || jloc < 0 || kloc < 0; ibreak = true end
   return ibreak
end

"Create unit vectors of field."
function normalize_field(iSize::T, jSize::T, ux, uy, dx, dy) where {T<:Integer}
   fx, fy = similar(ux), similar(uy)
   dxInv, dyInv = 1/dx, 1/dy
   @inbounds for i = 1:iSize*jSize
      uInv = 1.0 / sqrt((ux[i]*dxInv)^2 + (uy[i]*dyInv)^2)
      fx[i] = ux[i] * dxInv * uInv
      fy[i] = uy[i] * dyInv * uInv
   end
   return fx, fy
end

"""
    grid_interp!(x, y, field, ix, iy, xsize)

Interpolate a value at (x,y) in a field. `ix` and `iy` are indexes for x,y
locations (0-based). `xsize` is the size of field in X.
"""
grid_interp!(x, y, field::Array, ix, iy) =
   bilin_reg(x-ix, y-iy,
   field[ix+1, iy+1],
   field[ix+2, iy+1],
   field[ix+1, iy+2],
   field[ix+2, iy+2])

"""
    Euler!(iSize,jSize, maxstep, ds, xstart,ystart, xGrid,yGrid, ux,uy, x,y)

Simple 2D tracing using Euler's method. Super fast but not super accurate.
# Arguments
- `iSize::Int,jSize::Int`: grid size.
- `maxstep::Int`: max steps.
- `ds::Float64`: step size.
- `xstart::Float64, ystart::Float64`: starting location.
- `xGrid::Vector{Float64},yGrid::Vector{Float64}`: actual coord system.
- `ux::Array{Float64,2},uy::Array{Float64,2}`: field to trace through.
- `x::Vector{Float64},y::Vector{Float64}`: x, y of result stream.
"""
function Euler!(iSize, jSize, maxstep, ds, xstart, ystart, xGrid, yGrid, ux, uy,
   x, y)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   x[1] = (xstart-xGrid[1]) / dx
   y[1] = (ystart-yGrid[1]) / dy

   # Create unit vectors from full vector field
   f1, f2 = normalize_field(iSize, jSize, ux, uy, dx, dy)

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
      fx = grid_interp!(x[n], y[n], f1, ix, iy)
      fy = grid_interp!(x[n], y[n], f2, ix, iy)

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
   return nstep
end

"""
    RK4!(iSize,jSize, maxstep, ds, xstart,ystart, xGrid,yGrid, ux,uy, x,y)

Fast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and
constant step size `ds`.
"""
function RK4!(iSize, jSize, maxstep, ds, xstart, ystart, xGrid, yGrid, ux, uy,
   x, y)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   x[1] = (xstart-xGrid[1]) / dx
   y[1] = (ystart-yGrid[1]) / dy

   # Create unit vectors from full vector field
   fx, fy = normalize_field(iSize, jSize, ux, uy, dx, dy)

   nstep = 0
   # Perform tracing using RK4
   for n = 1:maxstep-1
      # See Euler's method for more descriptive comments.
      # SUBSTEP #1
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f1x = grid_interp!(x[n], y[n], fx, ix, iy)
      f1y = grid_interp!(x[n], y[n], fy, ix, iy)
      if isnan(f1x) || isnan(f1y) || isinf(f1x) || isinf(f1y)
         nstep = n; break
      end
      # SUBSTEP #2
      xpos = x[n] + f1x*ds/2.0
      ypos = y[n] + f1y*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f2x = grid_interp!(xpos, ypos, fx, ix, iy)
      f2y = grid_interp!(xpos, ypos, fy, ix, iy)

      if isnan(f2x) || isnan(f2y) || isinf(f2x) || isinf(f2y)
         nstep = n; break
      end
      # SUBSTEP #3
      xpos = x[n] + f2x*ds/2.0
      ypos = y[n] + f2y*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f3x = grid_interp!(xpos, ypos, fx, ix, iy)
      f3y = grid_interp!(xpos, ypos, fy, ix, iy)
      if isnan(f3x) || isnan(f3y) || isinf(f3x) || isinf(f3y)
         nstep = n; break
      end

      # SUBSTEP #4
      xpos = x[n] + f3x*ds
      ypos = y[n] + f3y*ds
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      if DoBreak(ix, iy, iSize, jSize); nstep = n; break end

      f4x = grid_interp!(xpos, ypos, fx, ix, iy)
      f4y = grid_interp!(xpos, ypos, fy, ix, iy)
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
   return nstep
end

"""
	 trace2d_rk4(fieldx, fieldy, xstart, ystart, gridx, gridy;
		 maxstep=20000, ds=0.01, gridType="meshgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of
the vector field. The field is integrated using Runge Kutta 4. Slower than
Euler, but more accurate. The higher accuracy allows for larger step sizes `ds`.
 Only valid for regular grid with coordinates `gridx`, `gridy`. If `gridx` and
`gridy` are not given, assume that `xstart` and `ystart` are normalized
coordinates (e.g., position in terms of array indices.???)
The field can be in both `meshgrid` (default) or `ndgrid` format.
Supporting `direction` for {"both","forward","backward"}.
"""
function trace2d_rk4(fieldx, fieldy, xstart, ystart, gridx, gridy;
   maxstep=20000, ds=0.01, gridType="meshgrid", direction="both")

   xt = Vector{eltype(fieldx)}(undef,maxstep) # output x
   yt = Vector{eltype(fieldy)}(undef,maxstep) # output y

   nx, ny = size(gridx)[1], size(gridy)[1]

   gx, gy = collect(gridx), collect(gridy)

   if gridType == "ndgrid" # ndgrid
      fx = fieldx
      fy = fieldy
   else # meshgrid
      fx = permutedims(fieldx)
      fy = permutedims(fieldy)
   end

   if direction == "forward"
      npoints = RK4!(nx,ny, maxstep, ds, xstart,ystart, gx,gy, fx,fy, xt,yt)
   elseif direction == "backward"
      npoints = RK4!(nx,ny, maxstep, ds, xstart,ystart, gx,gy, -fx,-fy, xt,yt)
   else
      n1 = RK4!(nx,ny,floor(Int,maxstep/2),ds,xstart,ystart,gx,gy,-fx,-fy,xt,yt)
      xt[n1:-1:1] = xt[1:n1]
      yt[n1:-1:1] = yt[1:n1]

      x2 = Vector{eltype(fieldx)}(undef,maxstep-n1)
      y2 = Vector{eltype(fieldx)}(undef,maxstep-n1)
      n2 = RK4!(nx,ny, maxstep-n1, ds, xstart,ystart, gx,gy, fx,fy, x2,y2)
      xt[n1+1:n1+n2-1] = x2[2:n2]
      yt[n1+1:n1+n2-1] = y2[2:n2]
      npoints = n1 + n2 - 1
   end

   return xt[1:npoints], yt[1:npoints]
end

"""
	 trace2d_eul(fieldx, fieldy, xstart, ystart, gridx, gridy;
		 maxstep=20000, ds=0.01, gridType="meshgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of
the vector field. The field is integrated using Euler's method. While this is
faster than rk4, it is less accurate. Only valid for regular grid with
coordinates `gridx`, `gridy`. If gridx and gridy are not given, assume that
`xstart` and `ystart` are normalized coordinates (e.g. position in terms of
array indices.)??? The field can be in both `meshgrid` (default) or `ndgrid`
format.
Supporting `direction` for {"both","forward","backward"}.
"""
function trace2d_eul(fieldx, fieldy, xstart, ystart, gridx, gridy;
   maxstep=20000, ds=0.01, gridType="meshgrid", direction="both")

   xt = Vector{eltype(fieldx)}(undef,maxstep) # output x
   yt = Vector{eltype(fieldy)}(undef,maxstep) # output y

   nx, ny = size(gridx)[1], size(gridy)[1]

   gx, gy = collect(gridx), collect(gridy)

   if gridType == "ndgrid" # ndgrid
      fx = fieldx
      fy = fieldy
   else # meshgrid
      fx = permutedims(fieldx)
      fy = permutedims(fieldy)
   end

   if direction == "forward"
      npoints = Euler!(nx,ny, maxstep, ds, xstart,ystart, gx,gy, fx,fy, xt,yt)
   elseif direction == "backward"
      npoints = Euler!(nx,ny, maxstep, ds, xstart,ystart, gx,gy, -fx,-fy, xt,yt)
   else
      n1 = Euler!(nx,ny, floor(Int,maxstep/2), ds, xstart,ystart, gx,gy,-fx,-fy,
         xt,yt)
      xt[n1:-1:1] = xt[1:n1]
      yt[n1:-1:1] = yt[1:n1]

      x2 = Vector{eltype(fieldx)}(undef,maxstep-n1)
      y2 = Vector{eltype(fieldx)}(undef,maxstep-n1)
      n2 = Euler!(nx,ny, maxstep-n1, ds, xstart,ystart, gx,gy, fx,fy, x2,y2)
      xt[n1+1:n1+n2-1] = x2[2:n2]
      yt[n1+1:n1+n2-1] = y2[2:n2]
      npoints = n1 + n2 - 1
   end

   return xt[1:npoints], yt[1:npoints]
end

trace2d(fieldx, fieldy, xstart, ystart, gridx, gridy; kwargs...) =
   trace2d_rk4(fieldx, fieldy, xstart, ystart, gridx, gridy; kwargs...)