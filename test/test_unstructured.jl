# Test on the idea of 0th order stream tracing.
# Assume the value is constant in each cell.

using FieldTracer, Meshes

"Check if tracing on mixed 2D unstructured grid passes."
function test_trace_unstructured2D()

   # Define coordinates
   coords = zeros(2,9)
   coords[:,1] = [1, 1]
   coords[:,2] = [3, 1]
   coords[:,3] = [4, 1]
   coords[:,4] = [1, 2]
   coords[:,5] = [2, 2]
   coords[:,6] = [1, 3]
   coords[:,7] = [2, 3]
   coords[:,8] = [3, 3]
   coords[:,9] = [4, 3]

   points = Point2[coords[:,i] for i in 1:size(coords,2)]
   tris = connect.([(2,3,9), (5,8,7), (5,2,8), (2,9,8)], Triangle)
   quads = connect.([(1,2,5,4),(4,5,7,6)], Quadrangle)
   mesh = SimpleMesh(points, [tris; quads])

   vx = fill(0.5, 6)
   vy = fill(0.3, 6)
   start = [1.5, 1.5]
   
   xs, ys = trace(mesh, vx, vy, [start[1]], [start[2]])

   if length(xs[1]) == 4 && xs[1][4] ≈ 4.000005 && ys[1][4] ≈ 3.000003
      return true
   else
      return false
   end
end

## Visualization
#=
using PyPlot
fig, ax = plt.subplots()
ax.set_aspect("equal")
plot(coords[1,:], coords[2,:], marker="o", ls="", color="crimson")
plot(xs, ys, ".-")
=#