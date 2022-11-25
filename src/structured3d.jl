# 3D field tracing on a regular grid.

"""
    trilin_reg(x, y, z, Q)

Trilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1)
Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.
"""
function trilin_reg(x::T, y::T, z::T, Q::Vector{U}) where {T<:Real, U<:Number}
   oneT = one(T)
   fout =
      Q[1]*(oneT-x)*(oneT-y)*(oneT-z) +
      Q[2]* x *    (oneT-y)*(oneT-z) +
      Q[3]* y *    (oneT-x)*(oneT-z) +
      Q[4]* x * y * (oneT-z) +
      Q[5]*(oneT-x)*(oneT-y)*z +
      Q[6]* x *    (oneT-y)*z +
      Q[7]* y *    (oneT-x)*z +
      Q[8]* x * y * z
end

# Extension from 2d case
function DoBreak(iloc::Int, jloc::Int, kloc::Int, iSize::Int, jSize::Int, kSize::Int)
   ibreak = false
   if iloc ≥ iSize-1 || jloc ≥ jSize-1 || kloc ≥ kSize-1; ibreak = true end
   if iloc < 0 || jloc < 0 || kloc < 0; ibreak = true end
   ibreak
end

"Create unit vectors of field in normalized coordinates."
function normalize_field(ux::U, uy::U, uz::U, dx::V, dy::V, dz::V) where {U, V<:Real}
   fx, fy, fz = similar(ux), similar(uy), similar(uz)
   dxInv, dyInv, dzInv = 1/dx, 1/dy, 1/dz
   @inbounds for i = eachindex(ux)
      uInv = hypot(ux[i]*dxInv, uy[i]*dyInv, uz[i]*dzInv) |> inv
      fx[i] = ux[i] * dxInv * uInv
      fy[i] = uy[i] * dyInv * uInv
      fz[i] = uz[i] * dzInv * uInv
   end
   fx, fy, fz
end

"""
    grid_interp(x, y, z, field, ix, iy, iz, xsize, ysize)

Interpolate a value at (x,y,z) in a field. `ix`,`iy` and `iz` are indexes for x, y and z
locations (0-based). `xsize` and `ysize` are the sizes of field in X and Y.
"""
grid_interp(x::T, y::T, z::T, field::AbstractArray{U,3}, ix::Int, iy::Int, iz::Int) where
   {T<:Real, U<:Number} =
   trilin_reg(x-ix, y-iy, z-iz,
   [
   field[ix+1, iy+1, iz+1],
   field[ix+2, iy+1, iz+1],
   field[ix+1, iy+2, iz+1],
   field[ix+2, iy+2, iz+1],
   field[ix+1, iy+1, iz+2],
   field[ix+2, iy+1, iz+2],
   field[ix+1, iy+2, iz+2],
   field[ix+2, iy+2, iz+2]
   ])

"""
    euler(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast 3D tracing using Euler's method. It takes at most `maxstep` with step size `ds` tracing
the vector field given by `ux,uy,uz` starting from  `(startx,starty,startz)` in the
Cartesian grid specified by ranges `xGrid`, `yGrid` and `zGrid`.
Return footprints' coordinates in (`x`,`y`,`z`).
"""
function euler(maxstep::Int, ds::T, startx::T, starty::T, startz::T, xGrid, yGrid, zGrid,
   ux, uy, uz) where T<:AbstractFloat
   @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

   x = Vector{eltype(startx)}(undef, maxstep)
   y = Vector{eltype(starty)}(undef, maxstep)
   z = Vector{eltype(startz)}(undef, maxstep)

   iSize, jSize, kSize = size(xGrid,1), size(yGrid,1), size(zGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   dz = zGrid[2] - zGrid[1]
   x[1] = (startx - xGrid[1]) / dx
   y[1] = (starty - yGrid[1]) / dy
   z[1] = (startz - zGrid[1]) / dz

   # Create unit vectors from full vector field
   f1, f2, f3 = normalize_field(ux, uy, uz, dx, dy, dz)

   nstep = 0
   # Perform tracing using Euler's method
   for n = 1:maxstep-1
      # Find surrounding points
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])
      iz = floor(Int, z[n])

      # Break if we leave the domain
      if DoBreak(ix, iy, iz, iSize, jSize, kSize)
         nstep = n; break
      end

      # Interpolate unit vectors to current location
      fx = grid_interp(x[n], y[n], z[n], f1, ix,iy,iz)
      fy = grid_interp(x[n], y[n], z[n], f2, ix,iy,iz)
      fz = grid_interp(x[n], y[n], z[n], f3, ix,iy,iz)

      if any(isnan,[fx, fy, fz]) || any(isinf, [fx, fy, fz])
         nstep = n
         break
      end

      @muladd x[n+1] = x[n] + ds * fx
      @muladd y[n+1] = y[n] + ds * fy
      @muladd z[n+1] = z[n] + ds * fz

      nstep = n
   end

   # Convert traced points to original coordinate system.
   @inbounds for i = 1:nstep
      @muladd x[i] = x[i]*dx + xGrid[1]
      @muladd y[i] = y[i]*dy + yGrid[1]
      @muladd z[i] = z[i]*dz + zGrid[1]
   end
   x[1:nstep], y[1:nstep], z[1:nstep]
end


"""
    rk4(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and constant step
size `ds`. See also [`euler`](@ref).
"""
function rk4(maxstep::Int, ds::T, startx::T, starty::T, startz::T, xGrid, yGrid, zGrid,
   ux, uy, uz) where T<:AbstractFloat

   @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

   x = Vector{eltype(startx)}(undef, maxstep)
   y = Vector{eltype(starty)}(undef, maxstep)
   z = Vector{eltype(startz)}(undef, maxstep)

   iSize, jSize, kSize = size(xGrid,1), size(yGrid,1), size(zGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   dz = zGrid[2] - zGrid[1]
   x[1] = (startx - xGrid[1]) / dx
   y[1] = (starty - yGrid[1]) / dy
   z[1] = (startz - zGrid[1]) / dz

   # Create unit vectors from full vector field
   fx, fy, fz = normalize_field(ux, uy, uz, dx, dy, dz)

   nstep = 0
   # Perform tracing using RK4
   for n = 1:maxstep-1
      # SUBSTEP #1
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])
      iz = floor(Int, z[n])
      if DoBreak(ix, iy, iz, iSize, jSize, kSize); nstep = n; break end

      f1x = grid_interp(x[n],y[n],z[n], fx, ix,iy,iz)
      f1y = grid_interp(x[n],y[n],z[n], fy, ix,iy,iz)
      f1z = grid_interp(x[n],y[n],z[n], fz, ix,iy,iz)
      if any(isnan,[f1x, f1y, f1z]) || any(isinf, [f1x, f1y, f1z])
         nstep = n; break
      end
      # SUBSTEP #2
      xpos = x[n] + f1x*ds/2.0
      ypos = y[n] + f1y*ds/2.0
      zpos = z[n] + f1z*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      iz = floor(Int, zpos)
      if DoBreak(ix, iy, iz, iSize, jSize, kSize); nstep = n; break end

      f2x = grid_interp(xpos,ypos,zpos, fx, ix,iy,iz)
      f2y = grid_interp(xpos,ypos,zpos, fy, ix,iy,iz)
      f2z = grid_interp(xpos,ypos,zpos, fz, ix,iy,iz)
      if any(isnan,[f2x, f2y, f2z]) || any(isinf, [f2x, f2y, f2z])
         nstep = n; break
      end
      # SUBSTEP #3
      xpos = x[n] + f2x*ds/2.0
      ypos = y[n] + f2y*ds/2.0
      zpos = z[n] + f2z*ds/2.0
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      iz = floor(Int, zpos)
      if DoBreak(ix, iy, iz, iSize, jSize, kSize); nstep = n; break end

      f3x = grid_interp(xpos,ypos,zpos, fx, ix,iy,iz)
      f3y = grid_interp(xpos,ypos,zpos, fy, ix,iy,iz)
      f3z = grid_interp(xpos,ypos,zpos, fz, ix,iy,iz)
      if any(isnan,[f3x, f3y, f3z]) || any(isinf, [f3x, f3y, f3z])
         nstep = n; break
      end

      # SUBSTEP #4
      xpos = x[n] + f3x*ds
      ypos = y[n] + f3y*ds
      zpos = z[n] + f3z*ds
      ix = floor(Int, xpos)
      iy = floor(Int, ypos)
      iz = floor(Int, zpos)
      if DoBreak(ix, iy, iz, iSize, jSize, kSize); nstep = n; break end

      f4x = grid_interp(xpos,ypos,zpos, fx, ix,iy,iz)
      f4y = grid_interp(xpos,ypos,zpos, fy, ix,iy,iz)
      f4z = grid_interp(xpos,ypos,zpos, fz, ix,iy,iz)
      if any(isnan,[f4x, f4y, f4z]) || any(isinf, [f4x, f4y, f4z])
         nstep = n; break
      end

      # Perform the full step using all substeps
      @muladd x[n+1] = x[n] + ds/6.0 * (f1x + f2x*2.0 + f3x*2.0 + f4x)
      @muladd y[n+1] = y[n] + ds/6.0 * (f1y + f2y*2.0 + f3y*2.0 + f4y)
      @muladd z[n+1] = z[n] + ds/6.0 * (f1z + f2z*2.0 + f3z*2.0 + f4z)

      nstep = n
   end

   # Convert traced points to original coordinate system.
   @inbounds for i = 1:nstep
      @muladd x[i] = x[i]*dx + xGrid[1]
      @muladd y[i] = y[i]*dy + yGrid[1]
      @muladd z[i] = z[i]*dz + zGrid[1]
   end
   x[1:nstep], y[1:nstep], z[1:nstep]
end

"""
	 trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
       maxstep=20000, ds=0.01)

Given a 3D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Euler's method. Only valid for regular grid with
coordinates `gridx`, `gridy`, `gridz`.
The field can be in both `meshgrid` or `ndgrid` (default) format.
Supporting `direction` of {"both","forward","backward"}.
"""
function trace3d_euler(fieldx, fieldy, fieldz, startx::T, starty::T, startz::T,
   gridx, gridy, gridz; maxstep::Int=20000, ds::T=0.01, gridtype::String="ndgrid",
   direction::String="both") where T<:AbstractFloat

   if gridtype == "ndgrid"
      fx, fy, fz = fieldx, fieldy, fieldz
   else # meshgrid
      fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
   end

   if direction == "forward"
      xt, yt, zt = euler(maxstep, ds, startx,starty,startz, gridx,gridy,gridz, fx, fy, fz)
   elseif direction == "backward"
      xt, yt, zt = euler(maxstep, ds, startx,starty,startz, gridx,gridy,gridz, -fx,-fy,-fz)
   else
      x1, y1, z1 = euler(floor(Int, maxstep/2), ds, startx, starty, startz,
         gridx, gridy, gridz, -fx, -fy, -fz)
      blen = length(x1)
      x2, y2, z2 = euler(maxstep-blen, ds, startx, starty, startz,
         gridx, gridy, gridz, fx, fy, fz)
      # concatenate with duplicates removed
      xt = vcat(reverse!(x1), x2[2:end])
      yt = vcat(reverse!(y1), y2[2:end])
      zt = vcat(reverse!(z1), z2[2:end])
   end

   xt, yt, zt
end


"""
	 trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
       maxstep=20000, ds=0.01)

Given a 3D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Euler's method. Only valid for regular grid with
coordinates `gridx`, `gridy`, `gridz`.
The field can be in both `meshgrid` or `ndgrid` (default) format.
See also [`trace3d_euler`](@ref).
"""
function trace3d_rk4(fieldx, fieldy, fieldz, startx::T, starty::T, startz::T,
   gridx, gridy, gridz; maxstep::Int=20000, ds::T=0.01, gridtype::String="ndgrid",
   direction::String="both") where T<:AbstractFloat

   if gridtype == "ndgrid"
      fx, fy, fz = fieldx, fieldy, fieldz
   else # meshgrid
      fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
   end

   if direction == "forward"
      xt, yt, zt = rk4(maxstep, ds, startx,starty,startz, gridx,gridy,gridz, fx, fy, fz)
   elseif direction == "backward"
      xt, yt, zt = rk4(maxstep, ds, startx,starty,startz, gridx,gridy,gridz, -fx,-fy,-fz)
   else
      x1, y1, z1 = rk4(floor(Int,maxstep/2), ds, startx, starty, startz,
         gridx, gridy, gridz, -fx, -fy, -fz)
      blen = length(x1)
      x2, y2, z2 = rk4(maxstep-blen, ds, startx, starty, startz,
         gridx, gridy, gridz, fx, fy, fz)
      # concatenate with duplicates removed
      xt = vcat(reverse!(x1), x2[2:end])
      yt = vcat(reverse!(y1), y2[2:end])
      zt = vcat(reverse!(z1), z2[2:end])
   end

   xt, yt, zt
end

"""
    trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;
		 maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

See also [`trace3d_rk4`](@ref).
"""
function trace3d_euler(fieldx::F, fieldy::F, fieldz::F, startx::T, starty::T, startz::T,
   grid::CartesianGrid; kwargs...) where {F, T}

   gridmin = coordinates(minimum(grid))
   gridmax = coordinates(maximum(grid))
   Δx = spacing(grid)

   gridx = range(gridmin[1], gridmax[1], step=Δx[1])
   gridy = range(gridmin[2], gridmax[2], step=Δx[2])
   gridz = range(gridmin[3], gridmax[3], step=Δx[3])

   trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
      kwargs...)
end