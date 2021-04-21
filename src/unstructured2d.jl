# 2D Field tracing on a unstructured grid.

export getCellID, trace2d

const Δ = 100000. # distance to the far field point
const ϵ = 1e-5 # small perturbation

"""
	 trace2d(mesh::SimpleMesh, vx, vy, xstart, ystart;
       maxIter=1000, maxLen=1000.)

2D stream tracing on unstructured quadrilateral and triangular mesh.
"""
function trace2d(mesh::SimpleMesh, vx, vy, xstart, ystart;
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
         
         if mesh[cellID] isa Quadrangle
            nEdge = 4
         elseif mesh[cellID] isa Triangle
            nEdge = 3
         end

         ray = Segment(Pnow, Pfar)
         # Loop over edges
         for i = 1:nEdge
            j = i % nEdge + 1 
            P = Segment(mesh.points[nodes[i]], mesh.points[nodes[j]]) ∩ ray
            if P isa Point
               P⁺ = P + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
               cellIDNew = getCellID(mesh, P⁺)
               break
            elseif P isa Segment
               P⁺ = mesh.points[nodes[j]] + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
               cellIDNew = getCellID(mesh, P⁺)
               break
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

   xStream, yStream
end

"""
    getCellID(mesh::SimpleMesh, x, y)

Return cell ID on the unstructured mesh.
"""
function getCellID(mesh::SimpleMesh, point::Point2)
   for i = 1:length(mesh.connec)
      nodes = mesh.connec[i].list
      if mesh[i] isa Triangle
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