# 2D Field tracing on a unstructured grid.

export trace

const Δ = 100000. # distance to the far field point
const ϵ = 1e-5 # small perturbation

"""
	 trace(mesh::SimpleMesh, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)

2D stream tracing on unstructured quadrilateral and triangular mesh.
"""
function trace(mesh::SimpleMesh, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)

   xStream = [fill(xs, maxIter) for xs in xstart]
   yStream = [fill(ys, maxIter) for ys in ystart]
   nIter = fill(maxIter, length(xStream))

   @inbounds for iS in 1:length(xStream)
      Pnow = Point(xStream[iS][1], yStream[iS][1])
      P⁺ = Point(0.0f0, 0.0f0)
      cellID = getCellID(mesh, Pnow)
      cellIDNew = 0

      for it = 1:maxIter
         Pfar = Pnow + Vec2f(vx[cellID]*Δ, vy[cellID]*Δ)
         element = getelement(mesh, cellID)
         
         if mesh[cellID] isa Quadrangle
            nEdge = 4
         elseif mesh[cellID] isa Triangle
            nEdge = 3
         end

         ray = Segment(Pnow, Pfar)
         # Loop over edges
         for i = 1:nEdge
            j = i % nEdge + 1 
            P = Segment(element.vertices[i], element.vertices[j]) ∩ ray
            if P isa Point
               P⁺ = P + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
               cellIDNew = getCellID(mesh, P⁺)
               break
            elseif P isa Segment
               P⁺ = element.vertices[j] + Vec2f(vx[cellID]*ϵ, vy[cellID]*ϵ)
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

   @inbounds for i = 1:length(xStream)
      xStream[i] = xStream[i][1:nIter[i]]
      yStream[i] = yStream[i][1:nIter[i]]
   end

   xStream, yStream
end

"Return cell ID on the unstructured mesh."
function getCellID(mesh::SimpleMesh, point::Point2)
   for (i, element) in enumerate(elements(mesh))
      if point ∈ element
         return i
      end
   end
   return 0 # out of mesh boundary
end

"Return the `cellID`th element of the mesh."
function getelement(mesh, cellID)
   for (i, element) in enumerate(elements(mesh))
      if i == cellID
         return element
      end
   end
end