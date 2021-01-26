# Test on the idea of 0th order stream tracing.
# Assume in each cell the value is constant.
#
#
# Hongyang Zhou, hyzhou@umich.edu 03/29/2020
using PyPlot
using UnstructuredGrids
using UnstructuredGrids.RefCellGallery: SQUARE, TRIANGLE

include("StreamTrace.jl")
using .StreamTrace

#=
function getNodeState(mesh::Mesh, U)
   V, E = mesh.V, mesh.E
   Nv, Ne = size(V,1), size(E,1)
   UN, count = zeros(Nv), zeros(Int,Nv)
   for e in 1:Ne
      for i in 1:3
         n = E[e,i]
         UN[n] += U[e]; count[n] += 1
      end
   end
   UN ./= count
end
=#

# Define coordinates
coords = zeros(2,9)
coords[:,1] = [1,1]
coords[:,2] = [3,1]
coords[:,3] = [4,1]
coords[:,4] = [1,2]
coords[:,5] = [2,2]
coords[:,6] = [1,3]
coords[:,7] = [2,3]
coords[:,8] = [3,3]
coords[:,9] = [4,3]

# Define connectivity
connect = [1,2,5,4,2,3,9,4,5,7,6,5,8,7,5,2,8,2,9,8]
offsets = [1,      5,    8,      12,   15,   18,   21]

# Define cell types
refcells = [SQUARE, TRIANGLE]
types = [1,2,1,2,2,2]

# Generate the Umesh object
mesh = UGrid(connect,offsets,types,refcells,coords)

pointLocation(mesh, 2.5, 1.0)

#@time pointLocation(mesh, 2.5, 2.0)

# Add cell center data
vx = fill(0.5, 6)
vy = fill(0.3, 6)
# Two extreme case tests!
#start = [1.5, 1.5]
#=
vx = fill(0.5, 6)
vy = fill(0.5, 6)
start = [1.5, 1.5]
=#


xS, yS = streamtrace(mesh, vx, vy)

figure()
plot(xS,yS, ".-")
