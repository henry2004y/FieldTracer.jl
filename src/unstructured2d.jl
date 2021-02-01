using Meshes

export getCellID, trace2d

const Δ = 100000. # Distance to the far field point
const ϵ = 1e-5 # small perturbation

"""
	 trace2d(mesh::UGrid, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)

2D stream tracing in unstructured quadrilateral and triangular mesh.
"""
function trace2d(mesh::UnstructuredMesh, vx, vy, xstart, ystart;
   maxIter=1000, maxLen=1000.)

   xStream = [fill(xs, maxIter) for xs in xstart]
   yStream = [fill(ys, maxIter) for ys in ystart]
   nIter = fill(maxIter, length(xStream))

   for iS in 1:length(xStream)
      Pnow = Point(xStream[iS][1], yStream[iS][1])
      P⁺ = Point(0.0f0, 0.0f0)
      cellID = getCellID(mesh, Pnow)
      cellIDNew = 0

      for it = 1:maxIter
         Pfar = Pnow + Vec2f(vx[cellID]*Δ, vy[cellID]*Δ)
         nodes = mesh.connec[cellID].list
         
         if polytopetype(mesh.connec[cellID]) == Quadrangle
            nEdge = 4
         elseif polytopetype(mesh.connec[cellID]) == Triangle
            nEdge = 3
         end

         ray = Segment(Pnow, Pfar)
         # Loop over edges
         for i = 1:nEdge
            j = i % nEdge + 1 
            P = Segment(mesh.points[nodes[i]], mesh.points[nodes[j]]) ∩ ray
            if !isnothing(P)
               if typeof(P) == Point2
                  P⁺ = P + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
                  cellIDNew = getCellID(mesh, P⁺)
                  break
               else # line segment
                  P⁺ = mesh.points[nodes[j]] + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
                  cellIDNew = getCellID(mesh, P⁺)
                  break
               end
            end
         end

         Pnow = P⁺
         
         xStream[iS][it+1] = Pnow.coords[1]
         yStream[iS][it+1] = Pnow.coords[2]
         nIter[iS] = it+1

         if cellIDNew == 0 # hit the boundary
            break
         else
            cellID = cellIDNew
         end
      end
   end

   for i = 1:length(xStream)
      xStream[i] = xStream[i][1:nIter[i]]
      yStream[i] = yStream[i][1:nIter[i]]
   end

   return xStream, yStream
end

"""
    getCellID(mesh::UnstructuredMesh, x, y)

Return cell ID in the unstructured grid.
"""
function getCellID(mesh::UnstructuredMesh, point::Point2)
   for i = 1:length(mesh.connec)
      nodes = mesh.connec[i].list
      if polytopetype(mesh.connec[i]) == Triangle
         if point ∈ Triangle(mesh.points[nodes[1]], mesh.points[nodes[2]],
            mesh.points[nodes[3]])
            return i
         end
      else # Quadrangle
         if point ∈ Quadrangle(mesh.points[nodes[1]], mesh.points[nodes[2]],
            mesh.points[nodes[3]], mesh.points[nodes[4]])
            return i
         end
      end
   end
   return 0 # out of mesh boundary
end