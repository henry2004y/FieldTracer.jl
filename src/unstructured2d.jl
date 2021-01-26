module Unstructured

using LinearAlgebra: dot
using UnstructuredGrids
using UnstructuredGrids.RefCellGallery: SQUARE, TRIANGLE

export pointLocation, streamtrace

const Δ = 100000. # Distance to the far field point
const ϵ = 1e-5 # small perturbation

"This function should be merged into `UnstructuredGrid`!"
function find_edge_cells(mesh::UGrid)
   nCell = length(mesh.celltypes)
   cell_to_edges = generate_cell_to_faces(1,mesh)
   nEdge = maximum(cell_to_edges.list)
   edge_to_cells = zeros(Int,2,nEdge)
   isEmpty = fill(true, nEdge)
   for iC = 1:nCell
      index_ = mesh.cells.ptrs[iC]:mesh.cells.ptrs[iC+1]-1
      cellnodes = mesh.cells.list[index_]
      cellpoints = mesh.coordinates[:,cellnodes]
      for iE = 1:size(cellpoints,2)
         iEdge = cell_to_edges.list[cell_to_edges.ptrs[iC]+iE-1]
         if isEmpty[iEdge]
            edge_to_cells[1,iEdge] = iC
            isEmpty[iEdge] = false
         else
            edge_to_cells[2,iEdge] = iC
         end
      end
   end
   return cell_to_edges, edge_to_cells
end


"""
    isPointInTriangle(P::Vector{T}, A::Vector{T}, B::Vector{T}, C::Vector{T})

Check if a given 2D point locates inside a triangle. Supports only floating
point representations.
"""
function isPointInTriangle(P::Vector{T},
   A::Vector{T}, B::Vector{T}, C::Vector{T}) where T<:AbstractFloat
   # Compute vectors
   v0 = C - A
   v1 = B - A
   v2 = P - A

   # Compute dot products
   dot00 = dot(v0, v0)
   dot01 = dot(v0, v1)
   dot02 = dot(v0, v2)
   dot11 = dot(v1, v1)
   dot12 = dot(v1, v2)

   # Compute barycentric coordinates
   invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
   u = (dot11 * dot02 - dot01 * dot12) * invDenom
   v = (dot00 * dot12 - dot01 * dot02) * invDenom

   # Check if point is in triangle
   return (u >= 0) && (v >= 0) && (u + v < 1)
end

"""
    isPointInTriangle(P::Vector{T}, Tr::Array{T,2})

Check if a given 2D point locates inside a triangle. Supports only floating
point representations.
"""
function isPointInTriangle(P::Vector{T}, Tr::Array{T,2}) where T<:AbstractFloat
   # Compute vectors
   v0 = @views Tr[:,3] - Tr[:,1]
   v1 = @views Tr[:,2] - Tr[:,1]
   v2 = @views P - Tr[:,1]

   # Compute dot products
   dot00 = dot(v0, v0)
   dot01 = dot(v0, v1)
   dot02 = dot(v0, v2)
   dot11 = dot(v1, v1)
   dot12 = dot(v1, v2)

   # Compute barycentric coordinates
   invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
   u = (dot11 * dot02 - dot01 * dot12) * invDenom
   v = (dot00 * dot12 - dot01 * dot02) * invDenom

   # Check if point is in triangle
   return (u >= 0) && (v >= 0) && (u + v < 1)
end


"""
    isPointInQuad(P::Vector{T}, Quad::Array{T,2})

Check if a given 2D point locates inside a quadrilateral. Supports only floating
point representations.
"""
function isPointInQuad(P::Vector{T}, Quad::Array{T,2}) where T<:AbstractFloat
   Tr1 = Quad[:,1:3]
   Tr2 = Quad[:,[2,3,1]]

   return isPointInTriangle(P, Tr1) || isPointInTriangle(P, Tr2)
end


"""
    pointLocation(mesh::UGrid, P)

Return cell ID in the unstructured grid `mesh` where 2D point `P` locates.
"""
function pointLocation(mesh::UGrid, P)
   x, y = P
   pointLocation(mesh, x, y)
end

"""
    pointLocation(mesh::UGrid, x, y)

Return cell ID in the unstructured grid `mesh` where 2D point `[x,y]` locates.
"""
function pointLocation(mesh::UGrid, x, y)
   nCell = length(mesh.celltypes)
   xy = [x,y]
   iCell = Inf
   for i in 1:nCell
      index_ = mesh.cells.ptrs[i]:mesh.cells.ptrs[i+1]-1
      cellnodes = mesh.cells.list[index_]
      if mesh.celltypes[i] == 1
         if isPointInQuad(xy, mesh.coordinates[:,cellnodes])
            iCell = i; break
         end
      elseif mesh.celltypes[i] == 2
         if isPointInTriangle(xy, mesh.coordinates[:,cellnodes])
            iCell = i; break
         end
      end
   end
   return iCell
end

"""
	 onSegment(p::Vector{T}, q::Vector{T}, r::Vector{T})

Given three colinear points `p`, `q`, and `r`, the function checks if
point `q` lies on line segment 'pr'.
"""
function onSegment(p::Vector{T}, q::Vector{T}, r::Vector{T}) where
   T<:AbstractFloat

   if (q[1] ≤ max(p[1], r[1]) && q[1] ≥ min(p[1], r[1]) &&
      q[2] ≤ max(p[2], r[2]) && q[2] ≥ min(p[2], r[2]))
      return true
   else
      return false
   end

end


"""
	 orientation(p::Vector{T}, q::Vector{T}, r::Vector{T})

Find orientation of ordered triplet (`p`, `q`, `r`).
The function returns following values:
`0` => colinear
`1` => clockwise
`2` => counterclockwise
"""
function orientation(p::Vector{T}, q::Vector{T}, r::Vector{T}) where
   T<:AbstractFloat

   val = (q[2] - p[2])*(r[1] - q[1]) - (q[1] - p[1])*(r[2] - q[2])

   if val == 0
      return 0 # colinear
   else
      return val > 0 ? 1 : 2 # clockwise or counter-clockwise
   end
end


"""
	 doIntersect(p1::Vector{T},q1::Vector{T},p2::Vector{T}, q2::Vector{T})

Check if line segment `p1`-`q1` intersects with `p2`-`q2`.
"""
function doIntersect(p1::Vector{T},q1::Vector{T},
   p2::Vector{T}, q2::Vector{T}) where T<:AbstractFloat
   # Find the four orientations needed for general and special cases
   o1 = orientation(p1, q1, p2)
   o2 = orientation(p1, q1, q2)
   o3 = orientation(p2, q2, p1)
   o4 = orientation(p2, q2, q1)

   # General case
   if o1 != o2 && o3 != o4 return true end

   # Special Cases
   # p1, q1 and p2 are colinear and p2 lies on segment p1q1
   if o1 == 0 && onSegment(p1, p2, q1) return true end

   # p1, q1 and p2 are colinear and q2 lies on segment p1q1
   if o2 == 0 && onSegment(p1, q2, q1) return true end

   # p2, q2 and p1 are colinear and p1 lies on segment p2q2
   if o3 == 0 && onSegment(p2, p1, q2) return true end

   # p2, q2 and q1 are colinear and q1 lies on segment p2q2
   if o4 == 0 && onSegment(p2, q1, q2) return true end

   return false # Doesn't fall in any of the above cases
end


"""
	 streamtrace(mesh::UGrid, vx, vy, xstart, ystart;
       MaxIter=1000, MaxLength=1000.)

2D stream tracing in unstructured quadrilateral and triangular mesh.
The code still needs to be polished.
"""
function streamtrace(mesh::UGrid, vx, vy, xstart, ystart;
   MaxIter=1000, MaxLength=1000.)

   xStream = [[xs] for xs in xstart]
   yStream = [[ys] for ys in ystart]

   cell_to_edges, edge_to_cells = find_edge_cells(mesh)

   for iS in 1:length(xStream)
      xNow = [xStream[iS][1], yStream[iS][1]]
      cellID = pointLocation(mesh, xNow[1], xNow[2])

      for it = 1:MaxIter
         xFar = [xNow[1]+vx[cellID]*Δ, xNow[2]+vy[cellID]*Δ]

         index_ = mesh.cells.ptrs[cellID]:mesh.cells.ptrs[cellID+1]-1
         cellnodes = mesh.cells.list[index_]
         cellpoints = mesh.coordinates[:,cellnodes]
         cells = [0, 0]
         
         if mesh.refcells[mesh.celltypes[cellID]].vtkid == SQUARE.vtkid
            if doIntersect(cellpoints[:,1],cellpoints[:,2],xNow,xFar) # 1st edge
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,1] != cellpoints[1,2]
                  a₁ = (cellpoints[2,2]-cellpoints[2,1])/(cellpoints[1,2]-cellpoints[1,1])
                  b₁ = cellpoints[2,1] - a₁*cellpoints[1,1]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,1]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            elseif doIntersect(cellpoints[:,2],cellpoints[:,3],xNow,xFar) # 2nd
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]+1]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,2] != cellpoints[1,3]
                  a₁ = (cellpoints[2,3]-cellpoints[2,2])/(cellpoints[1,3]-cellpoints[1,2])
                  b₁ = cellpoints[2,2] - a₁*cellpoints[1,2]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,2]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            elseif doIntersect(cellpoints[:,3],cellpoints[:,4],xNow,xFar) # 3rd
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]+2]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,3] != cellpoints[1,4]
                  a₁ = (cellpoints[2,4]-cellpoints[2,3])/(cellpoints[1,4]-cellpoints[1,3])
                  b₁ = cellpoints[2,3] - a₁*cellpoints[1,3]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,3]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            elseif doIntersect(cellpoints[:,4],cellpoints[:,1],xNow,xFar) # 4th
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]+3]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,1] != cellpoints[1,4]
                  a₁ = (cellpoints[2,1]-cellpoints[2,4])/(cellpoints[1,1]-cellpoints[1,4])
                  b₁ = cellpoints[2,4] - a₁*cellpoints[1,4]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,1]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            end
         elseif mesh.refcells[mesh.celltypes[cellID]].vtkid == TRIANGLE.vtkid
            if doIntersect(cellpoints[:,1],cellpoints[:,2],xNow,xFar) # 1st edge
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,1] != cellpoints[1,2]
                  a₁ = (cellpoints[2,2]-cellpoints[2,1])/(cellpoints[1,2]-cellpoints[1,1])
                  b₁ = cellpoints[2,1] - a₁*cellpoints[1,1]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,1]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            elseif doIntersect(cellpoints[:,2],cellpoints[:,3],xNow,xFar) # 2nd
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]+1]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,2] != cellpoints[1,3]
                  a₁ = (cellpoints[2,3]-cellpoints[2,2])/(cellpoints[1,3]-cellpoints[1,2])
                  b₁ = cellpoints[2,2] - a₁*cellpoints[1,2]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,2]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            elseif doIntersect(cellpoints[:,3],cellpoints[:,1],xNow,xFar) # 3rd
               iEdge = cell_to_edges.list[cell_to_edges.ptrs[cellID]+2]
               cells = edge_to_cells[:,iEdge]

               if cellpoints[1,1] != cellpoints[1,3]
                  a₁ = (cellpoints[2,1]-cellpoints[2,3])/(cellpoints[1,1]-cellpoints[1,3])
                  b₁ = cellpoints[2,3] - a₁*cellpoints[1,3]
                  a₂ = (xFar[2]-xNow[2])/(xFar[1]-xNow[1])
                  b₂ = xNow[2] - a₂*xNow[1]
                  x₀ = -(b₁-b₂)/(a₁-a₂)
                  y₀ = a₁*x₀ + b₁
               else
                  x₀ = cellpoints[1,1]
                  y₀ = xNow[2] + vy[cellID]/vx[cellID]*(x₀ - xNow[1])
               end
            end
         end

         if cells[2] == 0
            @info "hit the boundary!"
            break
         elseif cells[2] != cellID
            cellID = cells[2]
         else
            cellID = cells[1]
         end

         xNow = [x₀+vx[cellID]*ϵ, y₀+vy[cellID]*ϵ]

         append!(xStream[iS], xNow[1])
         append!(yStream[iS], xNow[2])
      end
   end

   return xStream, yStream
end


end
