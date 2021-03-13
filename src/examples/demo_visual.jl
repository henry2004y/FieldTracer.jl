# Visual demonstration of the tracing scheme in different analytical field.
using PyPlot, FieldTracer

include("../../test/utility/dipole.jl")

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

   x1, y1 = trace2d_rk4(vx, vy, xstart, ystart, x, y, ds=0.1)
   x2, y2 = trace2d_rk4(vx, vy, xstart, ystart, x, y, ds=0.5)
   x3, y3 = trace2d_rk4(vx, vy, xstart, ystart, x, y, ds=1.0)
   x4, y4 = trace2d_eul(vx, vy, xstart, ystart, x, y, ds=0.1)
   x5, y5 = trace2d_eul(vx, vy, xstart, ystart, x, y, ds=0.5)
   x6, y6 = trace2d_eul(vx, vy, xstart, ystart, x, y, ds=1.0)

   # analytical solution const = x*y
   c = xstart * ystart
   if IsSingle
      x_anly = 1.0f0:0.001f0:xmax
   else
      x_anly = 1.0:0.001:xmax
   end
   y_anly = c ./ x_anly

   fig = plt.figure(figsize=(8,6))
   ax1 = plt.subplot(111)
   ax1.plot(x_anly, y_anly, "r--", label="Analytic",linewidth=3.0)
   ax1.plot(x1, y1, "b",   label="RK4 ds=0.1", linewidth=1.5)
   ax1.plot(x2, y2, "b--", label="RK4 ds=0.5", linewidth=1.5)
   ax1.plot(x3, y3, "b:",  label="RK4 ds=1.0", linewidth=1.5)
   ax1.plot(x4, y4, "g", label="Euler ds=0.1", linewidth=.75)
   ax1.plot(x5, y5, "g--", label="Euler ds=0.5", linewidth=.75)
   ax1.plot(x6, y6, "g:",  label="Euler ds=1.0", linewidth=.75)
   ax1.legend(loc="upper right")
   if IsSingle
      ax1.set_title("Runge Kutta 4 vs Euler's: Asymptotic Field, Single Precision")
   else
      ax1.set_title("Runge Kutta 4 vs Euler's: Asymptotic Field, Double Precision")
   end
   ax1.set_xlabel("Normalized X Coordinate")
   ax1.set_ylabel("Normalized Y Coordinate")
   ax1.set_xlim([3, 30])
   ax1.set_ylim([0.25, 2.5 ])

   ax1.annotate("Euler's method diverges strongly\nalong curves except when"*
      " taking \nvery small steps.  RK4 is\nfar more accurate "*
      "for all dS.",
      xy=(10.5,0.83), xycoords="data", xytext=(9,.3),
      arrowprops=Dict(:fc=>"black",:shrink=>0.05),
      horizontalalignment="center")
   ax1.annotate("Tracing begins at x=1, y=10.",
      xy=(4.8,2.45), xycoords="data", xytext=(6,1.9),
      arrowprops=Dict(:fc=>"black",:shrink=>0.05))
   ax1.annotate("The trace continues until \n"*
      "x=200.  At that point, Euler \n"*
      "dS=0.1 and RK4 dS=1.0 converge\n"*
      " at the same point, despite \n"*
      "the 10X difference in step size.",
      xy=(29,.5), xycoords="data", xytext=(20,1),
      arrowprops=Dict(:fc=>"black",:shrink=>0.005),
      horizontalalignment="center")
end

"Trace field lines through a Earth like magnetic dipole field."
function test_trace_dipole()
   # Start by creating a field of unit vectors.
   x = -100.0:5.0:101.0
   y = -100.0:5.0:101.0

   bx, by = b_hat(x, y)

   fig2 = plt.figure(figsize=(12,6))
   fig2.subplots_adjust(wspace=0.15, left=0.08, right=0.94)
   ax2 = plt.subplot(121)
   ax3 = plt.subplot(322)
   ax4a= plt.subplot(347)
   ax4b= plt.subplot(348)
   ax5 = plt.subplot(326)
   ax2.quiver(x,y, bx, by, units="x", pivot="middle")
   ax3.quiver(x,y, bx, by, units="x", pivot="tip")
   ax4a.quiver(x,y,bx, by, units="x", pivot="tip")
   ax4b.quiver(x,y,bx, by, units="x", pivot="tip")
   ax5.quiver(x,y, bx, by, units="x", pivot="tip")

   # Trace through this field.
   xstart = 10.0 # ystart = 25.0
   ds = 0.1
   for ystart in 0.:5.:31.
      x1, y1 = trace2d_rk4(bx, by, xstart, ystart, x, y, ds=ds)
      l1 = ax2.plot(x1,y1,"b")[1]
      ax3.plot(x1,y1,"b"); ax4b.plot(x1,y1,"b")
      ax5.plot(x1,y1,"b"); ax4a.plot(x1,y1,"b")
      x2, y2 = trace2d_eul(bx, by, xstart, ystart, x, y, ds=ds)
      l2 = ax2.plot(x2,y2,"r")[1]
      ax3.plot(x2,y2,"r"); ax4b.plot(x2,y2,"r")
      ax5.plot(x2,y2,"r"); ax4a.plot(x2,y2,"r")
      x3, y3 = b_line(xstart, ystart, npoints=300)
      l3 = ax2.plot(x3,y3,"k--")[1]
      ax3.plot(x3,y3,"k--"); ax4b.plot(x3,y3,"k--")
      ax5.plot(x3,y3,"k--"); ax4a.plot(x3,y3,"k--")

      ax2.set_xlim([-2,  100])
      ax2.set_ylim([-30, 100])
      ax2.set_title("Full View")
      ax2.set_xlabel("Normalized X Coordinate")
      ax2.set_ylabel("Normalized Y Coordinate")
      ax2.legend((l1, l2, l3),("RK4", "Euler", "Analytical"), loc="upper left")

      ax3.set_title("Zoomed Views")
      ax3.set_xlim([8.5, 17.5])
      ax3.set_ylim([3, 33])
      goodpos = ax3.get_position()

      ax4a.set_xlim([20,30])
      ax4a.set_ylim([-12,12])
      pos = ax4a.get_position()
      pos.x0 = goodpos.x0
      pos.x1 = pos.x0 + (goodpos.x1-goodpos.x0)/2.0 -0.01
      ax4a.set_position(pos)

      ax4b.set_xlim([50,60])
      ax4b.set_ylim([-12,12])
      pos = ax4b.get_position()
      pos.x0 = goodpos.x0 + (goodpos.x1-goodpos.x0)/2.0 +0.01
      ax4b.set_position(pos)
      ax4b.yaxis.set_ticklabels([])

      ax5.set_xlim([1,7])
      ax5.set_ylim([-7,-3])
      ax5.set_xlabel("Normalized X Coordinate")

      fig2.suptitle("RK4 vs Euler's Method: Dipole Field for dS="*
         string(round(ds,digits=3)))
   end
end