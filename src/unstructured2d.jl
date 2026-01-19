# 2D Field tracing on a unstructured grid.

const Δ = 100000.0 # distance to the far field point
const ϵ = 1.0e-5 # small perturbation

"""
	 trace(mesh::SimpleMesh, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)

2D stream tracing on unstructured quadrilateral and triangular mesh.
"""
function trace(
        mesh::SimpleMesh, vx::Vector{TV}, vy::Vector{TV},
        xstart::Vector{TX}, ystart::Vector{TX}; maxIter::Int = 1000, maxLen::Float64 = 1000.0,
    ) where
    {TV <: AbstractFloat, TX <: AbstractFloat}
    xStream = [fill(xs, maxIter) for xs in xstart]
    yStream = [fill(ys, maxIter) for ys in ystart]
    nIter = fill(maxIter, length(xStream))

    # Build spatial index
    centroids = [centroid(e) for e in elements(mesh)]
    k_search = min(length(mesh), 10)
    searcher = KNearestSearch(centroids, k_search)

    @inbounds for iS in eachindex(xStream)
        Pnow = Point(xStream[iS][1], yStream[iS][1])
        P⁺ = Point(zero(TX), zero(TX))
        cellID = getCellID(mesh, Pnow, searcher)
        cellIDNew = 0

        for it in 1:maxIter
            Pfar = Pnow + Vec(TX(vx[cellID] * Δ), TX(vy[cellID] * Δ))
            element = getelement(mesh, cellID)

            if mesh[cellID] isa Quadrangle
                nEdge = 4
            elseif mesh[cellID] isa Triangle
                nEdge = 3
            end

            ray = Segment(Pnow, Pfar)
            # Loop over edges
            for i in 1:nEdge
                j = i % nEdge + 1
                P = Segment(element.vertices[i], element.vertices[j]) ∩ ray
                if P isa Point
                    P⁺ = P + Vec(TX(vx[cellID] * ϵ), TX(vy[cellID] * ϵ))
                    cellIDNew = getCellID(mesh, P⁺, searcher)
                    break
                elseif P isa Segment
                    P⁺ = element.vertices[j] + Vec(TX(vx[cellID] * ϵ), TX(vy[cellID] * ϵ))
                    cellIDNew = getCellID(mesh, P⁺, searcher)
                    break
                end
            end

            Pnow = P⁺

            xStream[iS][it + 1] = Pnow.coords.x.val
            yStream[iS][it + 1] = Pnow.coords.y.val
            nIter[iS] = it + 1

            if cellIDNew == 0 # hit the boundary
                break
            else
                cellID = cellIDNew
            end
        end
    end

    @inbounds @simd for i in eachindex(xStream)
        xStream[i] = xStream[i][1:nIter[i]]
        yStream[i] = yStream[i][1:nIter[i]]
    end

    return xStream, yStream
end

"Return cell ID on the unstructured mesh."
function getCellID(mesh::SimpleMesh, point::Point, searcher)
    # Find nearest centroids
    idxs = search(point, searcher)

    # Check validity among candidates
    for idx in idxs
        if point ∈ mesh[idx]
            return idx
        end
    end

    return 0 # out of mesh boundary
end

"Return the `cellID`th element of the mesh."
function getelement(mesh::SimpleMesh, cellID::Int)
    return mesh[cellID]
end
