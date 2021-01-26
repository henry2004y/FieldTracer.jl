# Functions for the generation of a dipole field.
#
# Modified from the original Python version by
# Los Alamos National Security, LLC. 2010

# mean value of B at the magnetic equator on the Earth's surface, [nT]
const B₀ = 31200. 

"""
	 b_mag(x, y)

Return the strength of the dipole magnetic field in nT for a position
`x`, `y`in units of planetary radius.
"""
function b_mag(x, y)
   r = @. √(x^2 + y^2)
   cosy = y./r
   B = @. B₀ * sqrt(1+3*cosy^2)/r^3
end

"""
	 b_mag(x, y, z)

Return the strength of the dipole magnetic field in nT for a position
`x`, `y`, `z` in units of planetary radius.
"""
function b_mag(x, y, z)
   r = @. √(x^2 + y^2 + z^2)
   cosθ = z ./ r
   B = @. B₀ * sqrt(1+3*cosθ^2)/r^3
end

"""
    b_hat(x, y)

Return two arrays, `bx` and `by`, corresponding to the direction of a dipole 
field at (`x`, `y`). The grid is organized in meshgrid (default) or ndgrid
format.
"""
function b_hat(x, y; gridType="meshgrid")
   if gridType == "meshgrid"
      xgrid = [i for _ in y, i in x]
      ygrid = [j for j in y, _ in x]
   else
      xgrid = [i for i in x, _ in y]
      ygrid = [j for _ in x, j in y]
   end

   r = @. sqrt(xgrid^2 + ygrid^2)
   cosy = ygrid./r
   sinx = xgrid./r

   denom = @. sqrt(1.0 + 3.0*cosy^2)

   br = @. 2.0 * cosy / denom
   bθ =          sinx ./ denom

   bx = @. br*sinx + bθ*cosy
   by = @. br*cosy - bθ*sinx

   return bx, by
end

"""
    b_hat(x, y, z)

Return the direction of a 3D dipole field at given locations.
The grid is organized in meshgrid or ndgrid (default) format.
"""
function b_hat(x, y, z; gridType="ndgrid")
   if gridType == "meshgrid"
      xgrid = [i for _ in y, i in x, _ in z]
      ygrid = [j for j in y, _ in x, _ in z]
      zgrid = [k for _ in y, _ in x, k in z]
   else
      xgrid = [i for i in x, _ in y, _ in z]
      ygrid = [j for _ in x, j in y, _ in z]
      zgrid = [k for _ in x, _ in y, k in z]
   end

   r = @. sqrt(xgrid^2 + ygrid^2 + zgrid^2)

   bx = @. 3*xgrid*zgrid/r^5
   by = @. 3*ygrid*zgrid/r^5
   bz = @. (3*zgrid^2 - r^2) / r^5

   return bx, by, bz
end

"""
	 b_line(x, y; npoint=30)

For a starting X, Y point return x and y vectors that trace the dipole field
line that passes through the given point.
"""
function b_line(x, y; npoints=30)
   r = sqrt(x^2 + y^2)
   try
      theta = atan(x/y)
   catch
      @warn "ZeroDivisionError"
      theta = pi/2.0
   end

   R = r/(sin(theta)^2)

   if x < 0
      theta = π:π/npoints:2.0π
   else
      theta = 0:π/npoints:π
   end

   r_vec = @. R * sin(theta)^2
   x_out = @. r_vec * sin(theta)
   y_out = @. r_vec * cos(theta)

   return x_out, y_out
end

"A quick test of the dipole field functions."
function test_dipole_2d()

   x = -100.0:5.0:101.0
   y = -100.0:5.0:101.0

   x_vec, y_vec = b_hat(x,y)

   fig = plt.figure(figsize=(10,8))
   ax1 = plt.subplot(111)

   ax1.quiver(x, y, x_vec, y_vec)

   for i in -120:10:121
      x,y = b_line(float(i), 0.0, npoints=100)
      ax1.plot(x, y, "b")
   end
   for theta in π/2.0:π/100.0:3.0π/2.0
      x = sin(theta)
      y = cos(theta)
      x, y = b_line(x, y, npoints=100)
      ax1.plot(x, y, "r")
   end
   ax1.set_xlim([-100, 100])
   ax1.set_ylim([-100, 100])
   plt.title("Unit vectors for an arbitrary dipole field")

   return true
end