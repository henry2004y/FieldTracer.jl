# 3D field tracing on a regular grid.

export trace3d, trace3d_eul

"""
    trilin_reg(x, y, z, Q)

Trilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1)
Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.
"""
function trilin_reg(x, y, z, Q)
   fout =
      Q[1]*(1.0-x)*(1.0-y)*(1.0-z) +
      Q[2]* x *    (1.0-y)*(1.0-z) +
      Q[3]* y *    (1.0-x)*(1.0-z) +
      Q[4]* x * y * (1.0-z) +
      Q[5]*(1.0-x)*(1.0-y)*z +
      Q[6]* x *    (1.0-y)*z +
      Q[7]* y *    (1.0-x)*z +
      Q[8]* x * y * z
end

# Extension from 2d case
function DoBreak(iloc, jloc, kloc, iSize, jSize, kSize)
   ibreak = false
   if iloc ≥ iSize-1 || jloc ≥ jSize-1 || kloc ≥ kSize-1; ibreak = true end
   if iloc < 0 || jloc < 0 || kloc < 0; ibreak = true end
   return ibreak
end

"Create unit vectors of field in normalized coordinates."
function normalize_field(ux, uy, uz, dx, dy, dz)
   fx, fy, fz = similar(ux), similar(uy), similar(uz)
   dxInv, dyInv, dzInv = 1/dx, 1/dy, 1/dz
   @inbounds for i = 1:length(ux)
      uInv = 1.0 / sqrt((ux[i]*dxInv)^2 + (uy[i]*dyInv)^2 + (uz[i]*dzInv)^2)
      fx[i] = ux[i] * dxInv * uInv
      fy[i] = uy[i] * dyInv * uInv
      fz[i] = uz[i] * dzInv * uInv
   end
   return fx, fy, fz
end

"""
    grid_interp!(x, y, z, field, ix, iy, iz, xsize, ysize)

Interpolate a value at (x,y,z) in a field. `ix`,`iy` and `iz` are indexes
for x,y and z locations (0-based). `xsize` and `ysize` are the sizes of field in
X and Y.
"""
grid_interp!(x, y, z, field, ix, iy, iz) =
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
    Euler!(maxstep, ds, xstart, ystart, zstart, xGrid, yGrid, zGrid, ux, uy, uz,
       x, y, z)

Simple 3D tracing using Euler's method.
# Arguments
- `maxstep::Int`: max steps.
- `ds::Float64`: step size.
- `xstart::Float64, ystart::Float64, zstart::Float64`: starting location.
- `xGrid::Array{Float64,2},yGrid::Array{Float64,2},zGrid::Array{Float64,2}`: actual coord system.
- `ux::Array{Float64,2},uy::Array{Float64,2},uz::Array{Float64,2}`: field to trace through.
- `x::Vector{Float64},y::Vector{Float64},z::Vector{Float64}`: x, y, z of result stream.
"""
function Euler!(maxstep, ds, xstart, ystart, zstart, xGrid, yGrid, zGrid,
   ux, uy, uz, x, y, z)

   @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

   iSize, jSize, kSize = size(xGrid,1), size(yGrid,1), size(zGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   dz = zGrid[2] - zGrid[1]
   x[1] = (xstart-xGrid[1]) / dx
   y[1] = (ystart-yGrid[1]) / dy
   z[1] = (zstart-zGrid[1]) / dz

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
      fx = grid_interp!(x[n], y[n], z[n], f1, ix,iy,iz)
      fy = grid_interp!(x[n], y[n], z[n], f2, ix,iy,iz)
      fz = grid_interp!(x[n], y[n], z[n], f3, ix,iy,iz)

      if any(isnan,[fx, fy, fz]) || any(isinf, [fx, fy, fz])
         nstep = n
         break
      end

      x[n+1] = x[n] + ds * fx
      y[n+1] = y[n] + ds * fy
      z[n+1] = z[n] + ds * fz

      nstep = n
   end

   # Return traced points to original coordinate system.
   for i = 1:nstep
      x[i] = x[i]*dx + xGrid[1]
      y[i] = y[i]*dy + yGrid[1]
      z[i] = z[i]*dz + zGrid[1]
   end
   return nstep
end


"""
    RK4!(maxstep, ds, xstart,ystart,zstart,
      xGrid,yGrid,zGrid, ux,uy,uz, x,y,z)

Fast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and
constant step size `ds`.
"""
function RK4!(iSize, jSize, kSize, maxstep, ds, xstart, ystart, zstart,
   xGrid, yGrid, zGrid, ux, uy, uz, x, y, z)

   @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

   iSize, jSize, kSize = size(xGrid,1), size(yGrid,1), size(zGrid,1)

   # Get starting points in normalized/array coordinates
   dx = xGrid[2] - xGrid[1]
   dy = yGrid[2] - yGrid[1]
   dz = zGrid[2] - zGrid[1]
   x[1] = (xstart-xGrid[1]) / dx
   y[1] = (ystart-yGrid[1]) / dy
   z[1] = (zstart-zGrid[1]) / dz

   # Create unit vectors from full vector field
   fx, fy, fz = normalize_field(ux, uy, uz, dx, dy, dz)

   nstep = 0
   # Perform tracing using RK4
   for n = 1:maxstep-1
      # See Euler's method for more descriptive comments.
      # SUBSTEP #1
      ix = floor(Int, x[n])
      iy = floor(Int, y[n])
      iz = floor(Int, z[n])
      if DoBreak(ix, iy, iz, iSize, jSize, kSize); nstep = n; break end

      f1x = grid_interp!(x[n],y[n],z[n], fx, ix,iy,iz)
      f1y = grid_interp!(x[n],y[n],z[n], fy, ix,iy,iz)
      f1z = grid_interp!(x[n],y[n],z[n], fz, ix,iy,iz)
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

      f2x = grid_interp!(xpos,ypos,zpos, fx, ix,iy,iz)
      f2y = grid_interp!(xpos,ypos,zpos, fy, ix,iy,iz)
      f2z = grid_interp!(xpos,ypos,zpos, fz, ix,iy,iz)
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

      f3x = grid_interp!(xpos,ypos,zpos, fx, ix,iy,iz)
      f3y = grid_interp!(xpos,ypos,zpos, fy, ix,iy,iz)
      f3z = grid_interp!(xpos,ypos,zpos, fz, ix,iy,iz)
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

      f4x = grid_interp!(xpos,ypos,zpos, fx, ix,iy,iz)
      f4y = grid_interp!(xpos,ypos,zpos, fy, ix,iy,iz)
      f4z = grid_interp!(xpos,ypos,zpos, fz, ix,iy,iz)
      if any(isnan,[f4x, f4y, f4z]) || any(isinf, [f4x, f4y, f4z])
         nstep = n; break
      end

      # Peform the full step using all substeps
      x[n+1] = x[n] + ds/6.0 * (f1x + f2x*2.0 + f3x*2.0 + f4x)
      y[n+1] = y[n] + ds/6.0 * (f1y + f2y*2.0 + f3y*2.0 + f4y)
      z[n+1] = z[n] + ds/6.0 * (f1z + f2z*2.0 + f3z*2.0 + f4z)

      nstep = n
   end

   # Return traced points to original coordinate system.
   for i = 1:nstep
      x[i] = x[i]*dx + xGrid[1]
      y[i] = y[i]*dy + yGrid[1]
      z[i] = z[i]*dz + zGrid[1]
   end
   return nstep
end

"""
	 trace3d_eul(fieldx, fieldy, fieldz, xstart, ystart, zstart, gridx, gridy,
       gridz; maxstep=20000, ds=0.01)

Given a 3D vector field, trace a streamline from a given point to the edge of
the vector field. The field is integrated using Euler's method. Only valid for
regular grid with coordinates `gridx`, `gridy`, `gridz`.
The field can be in both `meshgrid` or `ndgrid` (default) format.
"""
function trace3d_eul(fieldx, fieldy, fieldz, xstart, ystart, zstart, gridx,
   gridy, gridz; maxstep=20000, ds=0.01, gridType="ndgrid")

   xt = Vector{eltype(fieldx)}(undef,maxstep) # output x
   yt = Vector{eltype(fieldy)}(undef,maxstep) # output y
   zt = Vector{eltype(fieldz)}(undef,maxstep) # output z

   gx, gy, gz = collect(gridx), collect(gridy), collect(gridz)

   if gridType == "ndgrid"
      fx, fy, fz = fieldx, fieldy, fieldz
   else # meshgrid
      fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
   end

   npoints = Euler!(maxstep, ds, xstart,ystart,zstart, gx,gy,gz, fx,fy,fz,
      xt,yt,zt)

   return xt[1:npoints], yt[1:npoints], zt[1:npoints]
end

trace3d(fieldx, fieldy, fieldz, xstart, ystart, zstart, gridx, gridy, gridz) =
   trace3d_eul(fieldx, fieldy, fieldz, xstart, ystart, zstart,
   gridx, gridy, gridz)