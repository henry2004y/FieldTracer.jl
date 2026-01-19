# 2D Field tracing on a regular grid.

"""
	 bilin_reg(x, y, Q00, Q01, Q10, Q11)

Bilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1)
Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.
"""
function bilin_reg(x, y, Q00, Q10, Q01, Q11)
    mx = 1 - x
    my = 1 - y
    return fout =
        Q00 * mx * my +
        Q10 * x * my +
        Q01 * y * mx +
        Q11 * x * y
end

"""
	 grid_interp(x, y, field, ix, iy)

Interpolate a value at (x,y) in a field. `ix` and `iy` are indexes for x,y
locations (0-based).
"""
grid_interp(x, y, ix, iy, field::Array) =
    bilin_reg(
    x - ix, y - iy,
    field[ix + 1, iy + 1],
    field[ix + 2, iy + 1],
    field[ix + 1, iy + 2],
    field[ix + 2, iy + 2]
)

"""
	 DoBreak(iloc, jloc, iSize, jSize)

Check to see if we should break out of an integration.
"""
function DoBreak(iloc, jloc, iSize, jSize)
    ibreak = false
    if iloc ≥ iSize - 1 || jloc ≥ jSize - 1
        ibreak = true
    end
    if iloc < 0 || jloc < 0
        ibreak = true
    end
    return ibreak
end

"Create unit vectors of field."
function normalize_field(ux, uy, dx, dy)
    @warn "normalize_field is deprecated and will be removed in future versions."
    fx, fy = similar(ux), similar(uy)
    dxInv, dyInv = 1 / dx, 1 / dy
    @inbounds @simd for i in eachindex(ux)
        uInv = hypot(ux[i] * dxInv, uy[i] * dyInv) |> inv
        fx[i] = ux[i] * dxInv * uInv
        fy[i] = uy[i] * dyInv * uInv
    end

    return fx, fy
end

"""
    euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast 2D tracing using Euler's method. It takes at most `maxstep` with step size `ds`
tracing the vector field given by `ux,uy` starting from `(startx,starty)` in the Cartesian
grid specified by ranges `xGrid` and `yGrid`. Step size is in physical unit.
Return footprints' coordinates in (`x`, `y`).
"""
function euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
    @assert size(ux) == size(uy) "field array sizes must be equal!"

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)

    iSize, jSize = size(xGrid, 1), size(yGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    inv_dx, inv_dy = inv(dx), inv(dy)
    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy

    nstep = 0
    # Perform tracing using Euler's method
    @inbounds for n in 1:(maxstep - 1)
        # Find surrounding points
        ix = floor(Int, x[n])
        iy = floor(Int, y[n])

        # Break if we leave the domain
        if DoBreak(ix, iy, iSize, jSize)
            nstep = n
            break
        end

        # Interpolate field to current location
        fx = grid_interp(x[n], y[n], ix, iy, ux)
        fy = grid_interp(x[n], y[n], ix, iy, uy)

        if isnan(fx) || isnan(fy) || isinf(fx) || isinf(fy)
            nstep = n
            break
        end

        # Update the location
        # The velocity in the index space is v_i = v_physical / dx
        # The step size in the index space is ds_i = v_i / |v_physical| * ds
        # x[n+1] = x[n] + ds_i = x[n] + fx / |v| / dx * ds

        v_mag = hypot(fx, fy)

        # Stop if the field is zero
        if v_mag == 0
            nstep = n
            break
        end

        v_inv = inv(v_mag)

        @muladd x[n + 1] = x[n] + ds * fx * v_inv * inv_dx
        @muladd y[n + 1] = y[n] + ds * fy * v_inv * inv_dy

        nstep = n
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        @muladd x[i] = x[i] * dx + xGrid[1]
        @muladd y[i] = y[i] * dy + yGrid[1]
    end

    return x[1:nstep], y[1:nstep]
end

"""
    rk4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and constant step
size `ds`. See also [`euler`](@ref).
"""
function rk4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
    @assert size(ux) == size(uy) "field array sizes must be equal!"

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)

    iSize, jSize = size(xGrid, 1), size(yGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    inv_dx, inv_dy = inv(dx), inv(dy)
    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy

    nstep = 0
    # Perform tracing using RK4
    @inbounds for n in 1:(maxstep - 1)
        # SUBSTEP #1
        ix = floor(Int, x[n])
        iy = floor(Int, y[n])
        if DoBreak(ix, iy, iSize, jSize)
            nstep = n
            break
        end

        f1x = grid_interp(x[n], y[n], ix, iy, ux)
        f1y = grid_interp(x[n], y[n], ix, iy, uy)

        if isnan(f1x) || isnan(f1y) || isinf(f1x) || isinf(f1y)
            nstep = n
            break
        end

        v1_mag = hypot(f1x, f1y)
        if v1_mag == 0
            nstep = n
            break
        end
        v1_inv = inv(v1_mag)

        # Convert to index space velocity
        k1x = f1x * v1_inv * inv_dx
        k1y = f1y * v1_inv * inv_dy

        # SUBSTEP #2
        @muladd xpos = x[n] + k1x * ds * 0.5
        @muladd ypos = y[n] + k1y * ds * 0.5
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        if DoBreak(ix, iy, iSize, jSize)
            nstep = n
            break
        end

        f2x = grid_interp(xpos, ypos, ix, iy, ux)
        f2y = grid_interp(xpos, ypos, ix, iy, uy)

        if isnan(f2x) || isnan(f2y) || isinf(f2x) || isinf(f2y)
            nstep = n
            break
        end

        v2_mag = hypot(f2x, f2y)
        if v2_mag == 0
            nstep = n
            break
        end
        v2_inv = inv(v2_mag)

        k2x = f2x * v2_inv * inv_dx
        k2y = f2y * v2_inv * inv_dy

        # SUBSTEP #3
        @muladd xpos = x[n] + k2x * ds * 0.5
        @muladd ypos = y[n] + k2y * ds * 0.5
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        if DoBreak(ix, iy, iSize, jSize)
            nstep = n
            break
        end

        f3x = grid_interp(xpos, ypos, ix, iy, ux)
        f3y = grid_interp(xpos, ypos, ix, iy, uy)

        if isnan(f3x) || isnan(f3y) || isinf(f3x) || isinf(f3y)
            nstep = n
            break
        end

        v3_mag = hypot(f3x, f3y)
        if v3_mag == 0
            nstep = n
            break
        end
        v3_inv = inv(v3_mag)

        k3x = f3x * v3_inv * inv_dx
        k3y = f3y * v3_inv * inv_dy

        # SUBSTEP #4
        @muladd xpos = x[n] + k3x * ds
        @muladd ypos = y[n] + k3y * ds
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        if DoBreak(ix, iy, iSize, jSize)
            nstep = n
            break
        end

        f4x = grid_interp(xpos, ypos, ix, iy, ux)
        f4y = grid_interp(xpos, ypos, ix, iy, uy)

        if isnan(f4x) || isnan(f4y) || isinf(f4x) || isinf(f4y)
            nstep = n
            break
        end

        v4_mag = hypot(f4x, f4y)
        if v4_mag == 0
            nstep = n
            break
        end
        v4_inv = inv(v4_mag)

        k4x = f4x * v4_inv * inv_dx
        k4y = f4y * v4_inv * inv_dy

        # Peform the full step using all substeps
        @muladd x[n + 1] = x[n] + ds / 6 * (k1x + k2x * 2 + k3x * 2 + k4x)
        @muladd y[n + 1] = y[n] + ds / 6 * (k1y + k2y * 2 + k3y * 2 + k4y)

        nstep = n
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        @muladd x[i] = x[i] * dx + xGrid[1]
        @muladd y[i] = y[i] * dy + yGrid[1]
    end

    return x[1:nstep], y[1:nstep]
end

"""
    trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy;
       maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Runge Kutta 4. Slower than Euler, but more accurate.
The higher accuracy allows for larger step sizes `ds`. Step size is in same unit as the
grid coordinates. See also [`trace2d_euler`](@ref).
"""
function trace2d_rk4(
        fieldx, fieldy, startx, starty, gridx, gridy;
        maxstep = 20000, ds = 0.01, gridtype = "ndgrid", direction = "both"
    )
    @assert ndims(gridx) == 1 "Grid must be given in 1D range or vector!"

    if gridtype == "ndgrid"
        fx = fieldx
        fy = fieldy
    else # meshgrid
        fx = permutedims(fieldx)
        fy = permutedims(fieldy)
    end

    if direction == "forward"
        xt, yt = rk4(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
    elseif direction == "backward"
        xt, yt = rk4(maxstep, -ds, startx, starty, gridx, gridy, fx, fy)
    else
        x1, y1 = rk4(floor(Int, maxstep / 2), -ds, startx, starty, gridx, gridy, fx, fy)
        blen = length(x1)
        x2, y2 = rk4(maxstep - blen, ds, startx, starty, gridx, gridy, fx, fy)
        # concatenate with duplicates removed
        xt = vcat(reverse!(x1), x2[2:end])
        yt = vcat(reverse!(y1), y2[2:end])
    end

    return xt, yt
end

"""
    trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy;
       maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

Given a 2D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Euler's method, which is faster but less accurate than
RK4. Only valid for regular grid with coordinates' range `gridx` and `gridy`. Step size is
in same unit as the grid coordinates. The field can be in both `meshgrid` or
`ndgrid` (default) format. Supporting `direction` of {"both","forward","backward"}.
"""
function trace2d_euler(
        fieldx, fieldy, startx, starty, gridx, gridy;
        maxstep = 20000, ds = 0.01, gridtype = "ndgrid", direction = "both"
    )
    @assert ndims(gridx) == 1 "Grid must be given in 1D range or vector!"

    if gridtype == "ndgrid"
        fx = fieldx
        fy = fieldy
    else # meshgrid
        fx = permutedims(fieldx)
        fy = permutedims(fieldy)
    end

    if direction == "forward"
        xt, yt = euler(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
    elseif direction == "backward"
        xt, yt = euler(maxstep, -ds, startx, starty, gridx, gridy, fx, fy)
    else
        x1, y1 = euler(floor(Int, maxstep / 2), -ds, startx, starty, gridx, gridy, fx, fy)
        blen = length(x1)
        x2, y2 = euler(maxstep - blen, ds, startx, starty, gridx, gridy, fx, fy)
        # concatenate with duplicates removed
        xt = vcat(reverse!(x1), x2[2:end])
        yt = vcat(reverse!(y1), y2[2:end])
    end

    return xt, yt
end

"""
    trace2d_euler(fieldx, fieldy, startx, starty, grid::CartesianGrid; kwargs...)
"""
function trace2d_euler(
        fieldx, fieldy, startx, starty, grid::CartesianGrid;
        kwargs...
    )
    gridmin = coords(minimum(grid))
    gridmax = coords(maximum(grid))
    Δx = spacing(grid)

    gridx = range(gridmin.x.val, gridmax.x.val, step = Δx[1].val)
    gridy = range(gridmin.y.val, gridmax.y.val, step = Δx[2].val)

    return trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)
end
