# Test on the idea of 0th order stream tracing.
# Assume in each cell the value is constant.
#
# Hongyang Zhou, hyzhou@umich.edu 03/29/2020

using UnstructuredGrids
using UnstructuredGrids.RefCellGallery: SQUARE, TRIANGLE
using FieldTracer
import FieldTracer: Unstructured.streamtrace, Unstructured.getCellID

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

   # Define connectivity
   connect = [1,2,5,4,2,3,9,4,5,7,6,5,8,7,5,2,8,2,9,8]
   offsets = [1,      5,    8,      12,   15,   18,   21]

   # Define cell types
   refcells = [SQUARE, TRIANGLE]
   types = [1,2,1,2,2,2]

   # Generate the Umesh object
   mesh = UGrid(connect, offsets, types, refcells, coords)

   #writevtk(grid,"foo") # -> generates file "foo.vtu" 

   #getCellID(mesh, 2.5, 1.0)

   # Add cell center vector data
   vx = fill(0.5, 6)
   vy = fill(0.3, 6)
   start = [1.5, 1.5]

   xs, ys = streamtrace(mesh, vx, vy, [start[1]], [start[2]])

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