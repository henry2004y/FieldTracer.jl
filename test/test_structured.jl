# Test functions for tracing in different kinds of analytical field.

include("dipole.jl")

"""
    test_trace_asymptote(IsSingle=false)

Test streamline tracing by plotting vectors and associated streamlines through a
simple velocity field where Vx=x, Vy=-y. Support for single and double precision
data.
"""
function test_trace_asymptote(IsSingle=false)

   # Start by creating a velocity vector field.
   if IsSingle
      xmax, ymax = 200.0f0, 20.0f0
      x = -10.0f0:0.25f0:xmax
      y = -10.0f0:0.25f0:ymax
   else
      xmax, ymax = 200.0, 20.0
      x = -10.0:0.25:xmax
      y = -10.0:0.25:ymax
   end

   xgrid = [i for j in y, i in x]
   ygrid = [j for j in y, i in x]

   if IsSingle
      vx = xgrid * 1.0f0
      vy = ygrid * -1.0f0

      xstart = 1.0f0
      ystart = 10.0f0
   else
      vx = xgrid * 1.0
      vy = ygrid * -1.0

      xstart = 1.0
      ystart = 10.0
   end

   x1, y1 = trace2d_rk4(vx, vy, xstart, ystart, x, y, ds=0.1, gridType="meshgrid")
   x2, y2 = trace2d_eul(vx, vy, xstart, ystart, x, y, ds=0.1, gridType="meshgrid")

   # analytical solution const = x*y
   c = xstart * ystart

   if IsSingle
      if isapprox(x1[end]*y1[end], c, atol=1e-2) &&
         isapprox(x2[end]*y2[end], c, atol=0.13)
         return true
      else
         return false
      end
   else
      if isapprox(x1[end]*y1[end], c, atol=1e-5) &&
         isapprox(x2[end]*y2[end], c, atol=0.12)
         return true
      else
         return false
      end
   end
end

"Trace field lines through a Earth like magnetic dipole field."
function test_trace_dipole()

   # Start by creating a field of unit vectors.
   x = -100.0:5.0:101.0
   y = -100.0:5.0:101.0

   bx, by = b_hat(x, y)

   # Trace through this field.
   xstart = 10.0
   ystart = 25.0
   ds = 0.1
   
   x1, y1 = trace2d_rk4(bx, by, xstart, ystart, x, y, ds=ds, gridType="meshgrid")
   x2, y2 = trace2d_eul(bx, by, xstart, ystart, x, y, ds=ds, gridType="meshgrid")

   if length(x1) == 260 && isapprox(x1[end]*y1[end], 7462.42688, atol=1e-5) &&
      length(x2) == 262 && isapprox(x2[end]*y2[end], 7601.60590, atol=1e-5)
      return true
   else
      return false
   end
end