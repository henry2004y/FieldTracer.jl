# Analytic magnetic field topology in 2D.
#
# This demo shows the two types of classical magnetic field null point in 2D
# calculated from an analytic function.
# In 2D, the magnetic field potential can be written as
# Az(x,y) = B₀/2*(x^2/dx^2 ± y^2/dy^2)
# where
# -: X-point
# +: O-point
# and the corresponding field can be expressed as
# B(x,y,z) = ẑ × ∇Az + ẑ Bg
# where Bg is the guide field.
#
# Two different methods of computing B are provided:
# 1. hand-calculated form
# 2. auto-differentiation

using LinearAlgebra
using FieldTracer
using ForwardDiff
using StaticArrays
using PyPlot

"Hand-calculated o-type magnetic field from 2D magnetic potential."
function o_type(X, Y, B₀, Bg, dx, dy)
   Bx = [-B₀/dy^2 * y for y in Y, x in X]
   By = [B₀/dx^2 * x for y in Y, x in X]
   Bz = fill(Bg, size(Bx))

   Bx, By, Bz
end

"Hand-calculated x-type magnetic field from 2D magnetic potential."
function x_type(X, Y, B₀, Bg, dx, dy)
   Bx = [B₀/dy^2 * y for y in Y, x in X]
   By = [B₀/dx^2 * x for y in Y, x in X]
   Bz = fill(Bg, size(Bx))

   Bx, By, Bz
end

"Return o-type magnetic field from 2D potential using auto-differentiation."
function o_type_autodiff(X, Y, B₀, Bg, dx, dy)
   
   Az(x, y) = B₀/2*(x^2/dx^2 + y^2/dy^2)
   ∇Az(x, y, z) = ForwardDiff.gradient(p -> Az(p[1], p[2]), SVector(x,y,z))
   ẑ = SVector(0.0, 0.0, 1.0)
   B = Array{Float64,3}(undef, 3, length(X), length(Y))
   for j in axes(Y,1), i in axes(X,1)
      B[:,i,j] = ẑ × ∇Az(X[i], Y[j], 0.0)
      B[3,i,j] += Bg
   end
   B
end

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

## Main

# Parameters
Bg = 1.0 # Guide field, [nT]
B₀ = 1.0

fig, axs = plt.subplots(1, 2, figsize=(12,5))

Δ = 0.1           # grid interval
X = -5:Δ:5        # grid x range
Y = -5:Δ:5        # grid y range
dx, dy = 1.0, 1.0 # scale ratio of the analytic field

@time Bx, By, Bz = o_type(X, Y, B₀, Bg, dx, dy)
axs[1].streamplot(collect(X), collect(Y), Bx, By, minlength=0.1)
#@time B = o_type_autodiff(X, Y, B₀, Bg, dx, dy)
#axs[1].streamplot(collect(X), collect(Y), B[1,:,:]', B[2,:,:]', minlength=0.2)

xstart = 1.0:1.0:3.0 # seed starting point x
ystart = 1.0:1.0:3.0 # seed starting point y

for i in axes(xstart,1)
   x1, y1 = trace2d(Bx, By, xstart[i], ystart[i], X, Y; maxstep=3000, ds=0.1, gridType="meshgrid")
   line = axs[1].plot(x1, y1, "r--")
   add_arrow(line[1])
end
axs[1].plot(0, 0, color="k", marker="o", label="o-type null point")
axs[1].legend()
axs[1].axis("equal")


X = -5:Δ:5        # grid x range
Y = -3:Δ:3        # grid x range
dx, dy = 1.0, 0.1 # scale ratio of the analytic field

#Bx, By, Bz = x_type_autodiff(X, Y, B₀, Bg, dx, dy)
@time B = x_type_autodiff(X, Y, B₀, Bg, dx, dy)
axs[2].streamplot(collect(X), collect(Y), B[1,:,:]', B[2,:,:]')

xstart = [1.0, 3.0, 1.0] # seed starting point x
ystart = [0.0, 0.0, 1.0] # seed starting point y

for i in axes(xstart,1)
   x1, y1 = trace2d(B[1,:,:], B[2,:,:], xstart[i], ystart[i], X, Y; maxstep=3000, ds=0.1, gridType="ndgrid")
   axs[2].plot(x1, y1, "r--")
end
axs[2].plot(0, 0, color="k", marker="o", label="x-type null point")
axs[2].legend()
axs[2].axis("equal")

savefig("x_o_point.png", bbox_inches="tight")