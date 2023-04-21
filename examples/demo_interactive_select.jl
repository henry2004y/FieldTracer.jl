# Interactive seed selection in 2D.
#
# This script demonstrates how to select seed points interatively with PyPlot.

using LinearAlgebra
using FieldTracer
using ForwardDiff
using StaticArrays
using PyPlot

"Return x-type magnetic field from 2D potential using auto-differentiation."
function x_type_autodiff(X, Y, B₀, Bg, dx, dy)

   Az(x, y) = B₀/2*(x^2/dx^2 - y^2/dy^2)
   ∇Az(x, y, z) = ForwardDiff.gradient(p -> Az(p[1], p[2]), SVector(x,y,z))
   ẑ = SVector(0.0, 0.0, 1.0)
   B = Array{Float64,3}(undef, 3, length(X), length(Y))
   for j in axes(Y,1), i in axes(X,1)
      B[:,i,j] = ẑ × ∇Az(X[i], Y[j], 0.0)
      B[3,i,j] += Bg
   end
   B
end

"""
    add_arrow(line, size=12)

Add an arrow of `size` to the object `line` from Matplotlib. This requires importing PyPlot,
and only works for Line2D.
"""
function add_arrow(line, size=12)
   color = line.get_color()
   xdata, ydata = line.get_data()

   for i = 1:2
      start_ind = length(xdata) ÷ 3 * i
      end_ind = start_ind + 1
      line.axes.annotate("",
         xytext=(xdata[start_ind], ydata[start_ind]),
         xy=(xdata[end_ind], ydata[end_ind]),
         arrowprops=Dict("arrowstyle"=>"-|>", "color"=>color),
         size=size
      )
   end

end

function onclick(event, axs, Bx, By, X, Y)
   xs, ys = event.xdata, event.ydata
   u0 = [xs, ys]
   x1, y1 = trace(Bx, By, xs, ys, X, Y; maxstep=3000, ds=0.1, gridType="ndgrid")
   line = plot(x1, y1, "r--")
   add_arrow(line[1])
end

## Main

# Parameters
Bg = 1.0 # Guide field, [nT]
B₀ = 1.0

fig, axs = plt.subplots(figsize=(12,5))

Δ = 0.1           # grid interval
X = -5:Δ:5        # grid x range
Y = -3:Δ:3        # grid x range
dx, dy = 1.0, 0.1 # scale ratio of the analytic field

@time B = x_type_autodiff(X, Y, B₀, Bg, dx, dy)
axs.streamplot(collect(X), collect(Y), B[1,:,:]', B[2,:,:]')

xstart = [1.0, 3.0, 1.0] # seed starting point x
ystart = [0.0, 0.0, 1.0] # seed starting point y

for i in axes(xstart,1)
   x1, y1 = trace(B[1,:,:], B[2,:,:], xstart[i], ystart[i], X, Y;
      maxstep=3000, ds=0.1, gridType="ndgrid")
   axs.plot(x1, y1, "r--")
end
axs.plot(0, 0, color="k", marker="o", label="x-type null point")
axs.legend()
axs.axis("equal")

# Interactive method 1: draw a new streamline on mouse click
fig.canvas.mpl_connect("button_press_event",
   event -> onclick(event, axs, B[1,:,:], B[2,:,:], X, Y))

# Interactive mthod 2: select n seeds and then draw streamlines
#=
x = plt.ginput(3)
@show x
for i in eachindex(x)
   x1, y1 = trace2d(B[1,:,:], B[2,:,:], x[i][1], x[i][2], X, Y;
      maxstep=3000, ds=0.1, gridType="ndgrid")
   axs.plot(x1, y1, "r--")
end
=#