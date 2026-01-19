# 3D field tracing on a regular grid.

"""
    trilin_reg(x, y, z, Q)

Trilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1)
Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.
"""
function trilin_reg(
        x::T,
        y::T,
        z::T,
        Q000,
        Q100,
        Q010,
        Q110,
        Q001,
        Q101,
        Q011,
        Q111,
    ) where {T <: Real}
    oneT = one(T)
    mx = oneT - x
    my = oneT - y
    mz = oneT - z
    return fout =
        Q000 * mx * my * mz +
        Q100 * x * my * mz +
        Q010 * y * mx * mz +
        Q110 * x * y * mz +
        Q001 * mx * my * z +
        Q101 * x * my * z +
        Q011 * y * mx * z +
        Q111 * x * y * z
end

"""
    grid_interp(x, y, z, ix, iy, iz, field)

Interpolate a value at (x,y,z) in a field. `ix`,`iy` and `iz` are indexes for x, y and z
locations (0-based).
"""
grid_interp(
    x::T,
    y::T,
    z::T,
    ix::Int,
    iy::Int,
    iz::Int,
    field::AbstractArray{U, 3},
) where
{T <: Real, U <: Number} =
    trilin_reg(
    x - ix, y - iy, z - iz,
    field[ix + 1, iy + 1, iz + 1],
    field[ix + 2, iy + 1, iz + 1],
    field[ix + 1, iy + 2, iz + 1],
    field[ix + 2, iy + 2, iz + 1],
    field[ix + 1, iy + 1, iz + 2],
    field[ix + 2, iy + 1, iz + 2],
    field[ix + 1, iy + 2, iz + 2],
    field[ix + 2, iy + 2, iz + 2],
)

# Extension from 2d case
function isoutofdomain(iloc::Int, jloc::Int, kloc::Int, iSize::Int, jSize::Int, kSize::Int)
    ibreak = false
    if iloc ≥ iSize - 1 || jloc ≥ jSize - 1 || kloc ≥ kSize - 1
        ibreak = true
    end
    if iloc < 0 || jloc < 0 || kloc < 0
        ibreak = true
    end

    return ibreak
end

"""
    euler(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast 3D tracing using Euler's method. It takes at most `maxstep` with step size `ds` tracing
the vector field given by `ux,uy,uz` starting from  `(startx,starty,startz)` in the
Cartesian grid specified by ranges `xGrid`, `yGrid` and `zGrid`. `ds` is the step size in
physical unit.
Return footprints' coordinates in (`x`,`y`,`z`).
"""
@muladd function euler(
        maxstep::Int, ds::T, startx::T, starty::T, startz::T, xGrid, yGrid, zGrid,
        ux, uy, uz
    ) where {T}
    @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)
    z = Vector{eltype(startz)}(undef, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx, inv_dy, inv_dz = inv(dx), inv(dy), inv(dz)

    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy
    z[1] = (startz - zGrid[1]) * inv_dz

    nstep = 0
    # Perform tracing using Euler's method
    @inbounds for n in 1:(maxstep - 1)
        # Find surrounding points
        ix = floor(Int, x[n])
        iy = floor(Int, y[n])
        iz = floor(Int, z[n])

        # Break if we leave the domain
        if isoutofdomain(ix, iy, iz, iSize, jSize, kSize)
            nstep = n
            break
        end

        # Interpolate field to current location
        fx = grid_interp(x[n], y[n], z[n], ix, iy, iz, ux)
        fy = grid_interp(x[n], y[n], z[n], ix, iy, iz, uy)
        fz = grid_interp(x[n], y[n], z[n], ix, iy, iz, uz)

        if any(isnan, (fx, fy, fz)) || any(isinf, (fx, fy, fz))
            nstep = n
            break
        end

        v_mag = hypot(fx, fy, fz)
        if v_mag == 0
            nstep = n
            break
        end
        v_inv = inv(v_mag)

        x[n + 1] = x[n] + ds * fx * v_inv * inv_dx
        y[n + 1] = y[n] + ds * fy * v_inv * inv_dy
        z[n + 1] = z[n] + ds * fz * v_inv * inv_dz

        nstep = n
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        x[i] = x[i] * dx + xGrid[1]
        y[i] = y[i] * dy + yGrid[1]
        z[i] = z[i] * dz + zGrid[1]
    end

    return x[1:nstep], y[1:nstep], z[1:nstep]
end


"""
    rk4(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and constant step
size `ds`. See also [`euler`](@ref). `ds` is the step size in physical unit.
"""
@muladd function rk4(
        maxstep::Int, ds::T, startx::T, starty::T, startz::T, xGrid, yGrid, zGrid,
        ux, uy, uz
    ) where {T}

    @assert size(ux) == size(uy) == size(uz) "field array sizes must be equal!"

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)
    z = Vector{eltype(startz)}(undef, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx, inv_dy, inv_dz = inv(dx), inv(dy), inv(dz)

    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy
    z[1] = (startz - zGrid[1]) * inv_dz

    nstep = 0
    # Perform tracing using RK4
    @inbounds for n in 1:(maxstep - 1)
        # SUBSTEP #1
        ix = floor(Int, x[n])
        iy = floor(Int, y[n])
        iz = floor(Int, z[n])
        if isoutofdomain(ix, iy, iz, iSize, jSize, kSize)
            nstep = n
            break
        end

        f1x = grid_interp(x[n], y[n], z[n], ix, iy, iz, ux)
        f1y = grid_interp(x[n], y[n], z[n], ix, iy, iz, uy)
        f1z = grid_interp(x[n], y[n], z[n], ix, iy, iz, uz)

        if any(isnan, (f1x, f1y, f1z)) || any(isinf, (f1x, f1y, f1z))
            nstep = n
            break
        end

        v1_mag = hypot(f1x, f1y, f1z)
        if v1_mag == 0
            nstep = n
            break
        end
        v1_inv = inv(v1_mag)

        k1x = f1x * v1_inv * inv_dx
        k1y = f1y * v1_inv * inv_dy
        k1z = f1z * v1_inv * inv_dz

        # SUBSTEP #2
        xpos = x[n] + k1x * ds * 0.5
        ypos = y[n] + k1y * ds * 0.5
        zpos = z[n] + k1z * ds * 0.5
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        iz = floor(Int, zpos)
        if isoutofdomain(ix, iy, iz, iSize, jSize, kSize)
            nstep = n
            break
        end

        f2x = grid_interp(xpos, ypos, zpos, ix, iy, iz, ux)
        f2y = grid_interp(xpos, ypos, zpos, ix, iy, iz, uy)
        f2z = grid_interp(xpos, ypos, zpos, ix, iy, iz, uz)

        if any(isnan, (f2x, f2y, f2z)) || any(isinf, (f2x, f2y, f2z))
            nstep = n
            break
        end

        v2_mag = hypot(f2x, f2y, f2z)
        if v2_mag == 0
            nstep = n
            break
        end
        v2_inv = inv(v2_mag)

        k2x = f2x * v2_inv * inv_dx
        k2y = f2y * v2_inv * inv_dy
        k2z = f2z * v2_inv * inv_dz

        # SUBSTEP #3
        xpos = x[n] + k2x * ds * 0.5
        ypos = y[n] + k2y * ds * 0.5
        zpos = z[n] + k2z * ds * 0.5
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        iz = floor(Int, zpos)
        if isoutofdomain(ix, iy, iz, iSize, jSize, kSize)
            nstep = n
            break
        end

        f3x = grid_interp(xpos, ypos, zpos, ix, iy, iz, ux)
        f3y = grid_interp(xpos, ypos, zpos, ix, iy, iz, uy)
        f3z = grid_interp(xpos, ypos, zpos, ix, iy, iz, uz)

        if any(isnan, (f3x, f3y, f3z)) || any(isinf, (f3x, f3y, f3z))
            nstep = n
            break
        end

        v3_mag = hypot(f3x, f3y, f3z)
        if v3_mag == 0
            nstep = n
            break
        end
        v3_inv = inv(v3_mag)

        k3x = f3x * v3_inv * inv_dx
        k3y = f3y * v3_inv * inv_dy
        k3z = f3z * v3_inv * inv_dz

        # SUBSTEP #4
        xpos = x[n] + k3x * ds
        ypos = y[n] + k3y * ds
        zpos = z[n] + k3z * ds
        ix = floor(Int, xpos)
        iy = floor(Int, ypos)
        iz = floor(Int, zpos)
        if isoutofdomain(ix, iy, iz, iSize, jSize, kSize)
            nstep = n
            break
        end

        f4x = grid_interp(xpos, ypos, zpos, ix, iy, iz, ux)
        f4y = grid_interp(xpos, ypos, zpos, ix, iy, iz, uy)
        f4z = grid_interp(xpos, ypos, zpos, ix, iy, iz, uz)

        if any(isnan, (f4x, f4y, f4z)) || any(isinf, (f4x, f4y, f4z))
            nstep = n
            break
        end

        v4_mag = hypot(f4x, f4y, f4z)
        if v4_mag == 0
            nstep = n
            break
        end
        v4_inv = inv(v4_mag)

        k4x = f4x * v4_inv * inv_dx
        k4y = f4y * v4_inv * inv_dy
        k4z = f4z * v4_inv * inv_dz

        # Perform the full step using all substeps
        x[n + 1] = x[n] + ds / 6 * (k1x + k2x * 2 + k3x * 2 + k4x)
        y[n + 1] = y[n] + ds / 6 * (k1y + k2y * 2 + k3y * 2 + k4y)
        z[n + 1] = z[n] + ds / 6 * (k1z + k2z * 2 + k3z * 2 + k4z)

        nstep = n
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        x[i] = x[i] * dx + xGrid[1]
        y[i] = y[i] * dy + yGrid[1]
        z[i] = z[i] * dz + zGrid[1]
    end

    return x[1:nstep], y[1:nstep], z[1:nstep]
end

"""
    trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
       maxstep=20000, ds=0.01)

Given a 3D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Euler's method. Only valid for regular grid with
coordinates `gridx`, `gridy`, `gridz`.
The field can be in both `meshgrid` or `ndgrid` (default) format.
Supporting `direction` of {"both","forward","backward"}.
"""
function trace3d_euler(
        fieldx, fieldy, fieldz, startx::T, starty::T, startz::T,
        gridx, gridy, gridz; maxstep::Int = 20000, ds::T = 0.01, gridtype::String = "ndgrid",
        direction::String = "both"
    ) where {T}

    if gridtype == "ndgrid"
        fx, fy, fz = fieldx, fieldy, fieldz
    else # meshgrid
        fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
    end

    if direction == "forward"
        xt, yt, zt =
            euler(maxstep, ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    elseif direction == "backward"
        xt, yt, zt =
            euler(maxstep, -ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    else
        x1, y1, z1 = euler(
            floor(Int, maxstep / 2), -ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        blen = length(x1)
        x2, y2, z2 = euler(
            maxstep - blen, ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        # concatenate with duplicates removed
        xt = vcat(reverse!(x1), x2[2:end])
        yt = vcat(reverse!(y1), y2[2:end])
        zt = vcat(reverse!(z1), z2[2:end])
    end

    return xt, yt, zt
end


"""
    trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
       maxstep=20000, ds=0.01)

Given a 3D vector field, trace a streamline from a given point to the edge of the vector
field. The field is integrated using Euler's method. Only valid for regular grid with
coordinates `gridx`, `gridy`, `gridz`.
The field can be in both `meshgrid` or `ndgrid` (default) format.
See also [`trace3d_euler`](@ref).
"""
function trace3d_rk4(
        fieldx, fieldy, fieldz, startx::T, starty::T, startz::T,
        gridx, gridy, gridz; maxstep::Int = 20000, ds::T = 0.01, gridtype::String = "ndgrid",
        direction::String = "both"
    ) where {T}

    if gridtype == "ndgrid"
        fx, fy, fz = fieldx, fieldy, fieldz
    else # meshgrid
        fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
    end

    if direction == "forward"
        xt, yt, zt = rk4(maxstep, ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    elseif direction == "backward"
        xt, yt, zt =
            rk4(maxstep, -ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    else
        x1, y1, z1 = rk4(
            floor(Int, maxstep / 2), -ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        blen = length(x1)
        x2, y2, z2 = rk4(
            maxstep - blen, ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        # concatenate with duplicates removed
        xt = vcat(reverse!(x1), x2[2:end])
        yt = vcat(reverse!(y1), y2[2:end])
        zt = vcat(reverse!(z1), z2[2:end])
    end

    return xt, yt, zt
end

"""
    trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;
       maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

See also [`trace3d_rk4`](@ref).
"""
function trace3d_euler(
        fieldx::F, fieldy::F, fieldz::F, startx::T, starty::T, startz::T,
        grid::CartesianGrid; kwargs...
    ) where {F, T}
    gridmin = coords(minimum(grid))
    gridmax = coords(maximum(grid))
    Δx = spacing(grid)

    gridx = range(gridmin.x.val, gridmax.x.val, step = Δx[1].val)
    gridy = range(gridmin.y.val, gridmax.y.val, step = Δx[2].val)
    gridz = range(gridmin.z.val, gridmax.z.val, step = Δx[3].val)

    return trace3d_euler(
        fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
        kwargs...
    )
end

"""
    trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;
       maxstep=20000, ds=0.01, gridtype="ndgrid", direction="both")

See also [`trace3d_euler`](@ref).
"""
function trace3d_rk4(
        fieldx::F, fieldy::F, fieldz::F, startx::T, starty::T, startz::T,
        grid::CartesianGrid; kwargs...
    ) where {F, T}
    gridmin = coords(minimum(grid))
    gridmax = coords(maximum(grid))
    Δx = spacing(grid)

    gridx = range(gridmin.x.val, gridmax.x.val, step = Δx[1].val)
    gridy = range(gridmin.y.val, gridmax.y.val, step = Δx[2].val)
    gridz = range(gridmin.z.val, gridmax.z.val, step = Δx[3].val)

    return trace3d_rk4(
        fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;
        kwargs...
    )
end
