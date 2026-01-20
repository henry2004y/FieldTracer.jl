# 3D field tracing on a regular grid.

"""
    trilin_reg(x, y, z, Q)

Trilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1)
Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.
"""
@inline function trilin_reg(
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
@inline grid_interp(
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
@inline check_domain(ix, iy, iz, iSize, jSize, kSize) =
    (ix >= 0) & (ix < iSize - 1) & (iy >= 0) & (iy < jSize - 1) & (iz >= 0) & (iz < kSize - 1)

@inline check_validity(fx, fy, fz, v_mag) =
    (v_mag > 0) & !isnan(fx) & !isnan(fy) & !isnan(fz) & !isinf(fx) & !isinf(fy) & !isinf(fz)


@inline Base.@propagate_inbounds function _euler_step(cx, cy, cz, ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)
    ix = floor(Int, cx)
    iy = floor(Int, cy)
    iz = floor(Int, cz)

    in_domain = check_domain(ix, iy, iz, iSize, jSize, kSize)

    ix_c = max(0, min(ix, iSize - 2))
    iy_c = max(0, min(iy, jSize - 2))
    iz_c = max(0, min(iz, kSize - 2))

    fx = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, ux)
    fy = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, uy)
    fz = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, uz)

    v_mag = sqrt(fx^2 + fy^2 + fz^2)

    valid = in_domain & check_validity(fx, fy, fz, v_mag)

    v_inv = ifelse(valid, inv(v_mag), 0.0)

    dx = ifelse(valid, ds * fx * v_inv * inv_dx, 0.0)
    dy = ifelse(valid, ds * fy * v_inv * inv_dy, 0.0)
    dz = ifelse(valid, ds * fz * v_inv * inv_dz, 0.0)

    return cx + dx, cy + dy, cz + dz, valid
end

@inline Base.@propagate_inbounds function _rk4_step(cx, cy, cz, ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)
    # --- Substep 1 ---
    ix = floor(Int, cx); iy = floor(Int, cy); iz = floor(Int, cz)
    valid = check_domain(ix, iy, iz, iSize, jSize, kSize)
    ix_c, iy_c, iz_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2)), max(0, min(iz, kSize - 2))

    f1x = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, ux)
    f1y = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, uy)
    f1z = grid_interp(cx, cy, cz, ix_c, iy_c, iz_c, uz)
    v1_mag = sqrt(f1x^2 + f1y^2 + f1z^2)
    valid &= check_validity(f1x, f1y, f1z, v1_mag)
    v1_inv = ifelse(valid, inv(v1_mag), 0.0)

    k1x = ifelse(valid, f1x * v1_inv * inv_dx, 0.0)
    k1y = ifelse(valid, f1y * v1_inv * inv_dy, 0.0)
    k1z = ifelse(valid, f1z * v1_inv * inv_dz, 0.0)

    # --- Substep 2 ---
    px = cx + k1x * ds * 0.5
    py = cy + k1y * ds * 0.5
    pz = cz + k1z * ds * 0.5

    ix = floor(Int, px); iy = floor(Int, py); iz = floor(Int, pz)
    valid &= check_domain(ix, iy, iz, iSize, jSize, kSize)
    ix_c, iy_c, iz_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2)), max(0, min(iz, kSize - 2))

    f2x = grid_interp(px, py, pz, ix_c, iy_c, iz_c, ux)
    f2y = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uy)
    f2z = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uz)
    v2_mag = sqrt(f2x^2 + f2y^2 + f2z^2)
    valid &= check_validity(f2x, f2y, f2z, v2_mag)
    v2_inv = ifelse(valid, inv(v2_mag), 0.0)

    k2x = ifelse(valid, f2x * v2_inv * inv_dx, 0.0)
    k2y = ifelse(valid, f2y * v2_inv * inv_dy, 0.0)
    k2z = ifelse(valid, f2z * v2_inv * inv_dz, 0.0)

    # --- Substep 3 ---
    px = cx + k2x * ds * 0.5
    py = cy + k2y * ds * 0.5
    pz = cz + k2z * ds * 0.5

    ix = floor(Int, px); iy = floor(Int, py); iz = floor(Int, pz)
    valid &= check_domain(ix, iy, iz, iSize, jSize, kSize)
    ix_c, iy_c, iz_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2)), max(0, min(iz, kSize - 2))

    f3x = grid_interp(px, py, pz, ix_c, iy_c, iz_c, ux)
    f3y = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uy)
    f3z = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uz)
    v3_mag = sqrt(f3x^2 + f3y^2 + f3z^2)
    valid &= check_validity(f3x, f3y, f3z, v3_mag)
    v3_inv = ifelse(valid, inv(v3_mag), 0.0)

    k3x = ifelse(valid, f3x * v3_inv * inv_dx, 0.0)
    k3y = ifelse(valid, f3y * v3_inv * inv_dy, 0.0)
    k3z = ifelse(valid, f3z * v3_inv * inv_dz, 0.0)

    # --- Substep 4 ---
    px = cx + k3x * ds
    py = cy + k3y * ds
    pz = cz + k3z * ds

    ix = floor(Int, px); iy = floor(Int, py); iz = floor(Int, pz)
    valid &= check_domain(ix, iy, iz, iSize, jSize, kSize)
    ix_c, iy_c, iz_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2)), max(0, min(iz, kSize - 2))

    f4x = grid_interp(px, py, pz, ix_c, iy_c, iz_c, ux)
    f4y = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uy)
    f4z = grid_interp(px, py, pz, ix_c, iy_c, iz_c, uz)
    v4_mag = sqrt(f4x^2 + f4y^2 + f4z^2)
    valid &= check_validity(f4x, f4y, f4z, v4_mag)
    v4_inv = ifelse(valid, inv(v4_mag), 0.0)

    k4x = ifelse(valid, f4x * v4_inv * inv_dx, 0.0)
    k4y = ifelse(valid, f4y * v4_inv * inv_dy, 0.0)
    k4z = ifelse(valid, f4z * v4_inv * inv_dz, 0.0)

    # Update
    next_x = cx + ds / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    next_y = cy + ds / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    next_z = cz + ds / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)

    return next_x, next_y, next_z, valid
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
    @assert size(ux) == size(uy) == size(uz)

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)
    z = Vector{eltype(startz)}(undef, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx = inv(dx)
    inv_dy = inv(dy)
    inv_dz = inv(dz)
    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy
    z[1] = (startz - zGrid[1]) * inv_dz

    nstep = 1
    # Perform tracing using Euler's method
    @inbounds @fastmath for n in 1:(maxstep - 1)
        next_x, next_y, next_z, valid = _euler_step(x[n], y[n], z[n], ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)

        if !valid
            break
        end

        x[n + 1] = next_x
        y[n + 1] = next_y
        z[n + 1] = next_z
        nstep = n + 1
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
    euler_batch(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast 3D tracing of multiple particles using Euler's method with LoopVectorization.
Returns matrices `x`, `y`, `z` of size `(n_particles, maxstep)`.
"""
function euler_batch(
        maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid,
        ux, uy, uz
    )
    n_particles = length(startx)
    @assert length(starty) == length(startz) == n_particles
    @assert size(ux) == size(uy) == size(uz)

    x = Matrix{eltype(startx)}(undef, n_particles, maxstep)
    y = Matrix{eltype(starty)}(undef, n_particles, maxstep)
    z = Matrix{eltype(startz)}(undef, n_particles, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx = inv(dx)
    inv_dy = inv(dy)
    inv_dz = inv(dz)

    x0, y0, z0 = xGrid[1], yGrid[1], zGrid[1]

    @inbounds @fastmath @simd for i in 1:n_particles
        x[i, 1] = (startx[i] - x0) * inv_dx
        y[i, 1] = (starty[i] - y0) * inv_dy
        z[i, 1] = (startz[i] - z0) * inv_dz
    end

    nsteps = ones(Int, n_particles)

    for n in 1:(maxstep - 1)
        active_count = 0
        @inbounds @fastmath @simd for i in 1:n_particles
            cx = x[i, n]
            cy = y[i, n]
            cz = z[i, n]

            next_x, next_y, next_z, valid = _euler_step(cx, cy, cz, ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)

            # Update active count and nsteps
            active_count += valid
            nsteps[i] += valid

            x[i, n + 1] = next_x
            y[i, n + 1] = next_y
            z[i, n + 1] = next_z
        end

        if active_count == 0
            break
        end
    end

    @inbounds @fastmath @simd for n in 1:maxstep
        for i in 1:n_particles
            x[i, n] = x[i, n] * dx + x0
            y[i, n] = y[i, n] * dy + y0
            z[i, n] = z[i, n] * dz + z0
        end
    end

    return x, y, z, nsteps
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
    @assert size(ux) == size(uy) == size(uz)

    x = Vector{eltype(startx)}(undef, maxstep)
    y = Vector{eltype(starty)}(undef, maxstep)
    z = Vector{eltype(startz)}(undef, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx = inv(dx)
    inv_dy = inv(dy)
    inv_dz = inv(dz)
    x[1] = (startx - xGrid[1]) * inv_dx
    y[1] = (starty - yGrid[1]) * inv_dy
    z[1] = (startz - zGrid[1]) * inv_dz

    nstep = 1
    # Perform tracing using RK4's method
    @inbounds @fastmath for n in 1:(maxstep - 1)
        next_x, next_y, next_z, valid = _rk4_step(x[n], y[n], z[n], ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)

        if !valid
            break
        end

        x[n + 1] = next_x
        y[n + 1] = next_y
        z[n + 1] = next_z
        nstep = n + 1
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
    rk4_batch(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)

Fast 3D tracing of multiple particles using RK4 method with LoopVectorization.
Returns matrices `x`, `y`, `z` of size `(n_particles, maxstep)`.
"""
function rk4_batch(
        maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid,
        ux, uy, uz
    )
    n_particles = length(startx)
    @assert length(starty) == length(startz) == n_particles
    @assert size(ux) == size(uy) == size(uz)

    x = Matrix{eltype(startx)}(undef, n_particles, maxstep)
    y = Matrix{eltype(starty)}(undef, n_particles, maxstep)
    z = Matrix{eltype(startz)}(undef, n_particles, maxstep)

    iSize, jSize, kSize = size(xGrid, 1), size(yGrid, 1), size(zGrid, 1)

    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    dz = zGrid[2] - zGrid[1]
    inv_dx = inv(dx)
    inv_dy = inv(dy)
    inv_dz = inv(dz)
    x0, y0, z0 = xGrid[1], yGrid[1], zGrid[1]

    @inbounds @fastmath @simd for i in 1:n_particles
        x[i, 1] = (startx[i] - x0) * inv_dx
        y[i, 1] = (starty[i] - y0) * inv_dy
        z[i, 1] = (startz[i] - z0) * inv_dz
    end

    nsteps = ones(Int, n_particles)

    for n in 1:(maxstep - 1)
        active_count = 0
        @inbounds @fastmath @simd for i in 1:n_particles
            cx = x[i, n]
            cy = y[i, n]
            cz = z[i, n]

            next_x, next_y, next_z, valid = _rk4_step(cx, cy, cz, ux, uy, uz, ds, inv_dx, inv_dy, inv_dz, iSize, jSize, kSize)

            # Update active count and nsteps
            active_count += valid
            nsteps[i] += valid

            x[i, n + 1] = next_x
            y[i, n + 1] = next_y
            z[i, n + 1] = next_z
        end

        if active_count == 0
            break
        end
    end

    @inbounds @fastmath @simd for n in 1:maxstep
        for i in 1:n_particles
            x[i, n] = x[i, n] * dx + x0
            y[i, n] = y[i, n] * dy + y0
            z[i, n] = z[i, n] * dz + z0
        end
    end

    return x, y, z, nsteps
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

function trace3d_euler(
        fieldx, fieldy, fieldz, startx::AbstractVector, starty::AbstractVector, startz::AbstractVector,
        gridx, gridy, gridz; maxstep::Int = 20000, ds::T = 0.01, gridtype::String = "ndgrid",
        direction::String = "both"
    ) where {T}

    if gridtype == "ndgrid"
        fx, fy, fz = fieldx, fieldy, fieldz
    else # meshgrid
        fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
    end

    if direction == "forward"
        xt, yt, zt, _ =
            euler_batch(maxstep, ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    elseif direction == "backward"
        xt, yt, zt, _ =
            euler_batch(maxstep, -ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    else
        x1, y1, z1, _ = euler_batch(
            floor(Int, maxstep / 2), -ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        x2, y2, z2, _ = euler_batch(
            maxstep - size(x1, 2) + 1, ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )

        x1_rev = reverse(x1, dims = 2)
        y1_rev = reverse(y1, dims = 2)
        z1_rev = reverse(z1, dims = 2)

        xt = hcat(x1_rev, x2[:, 2:end])
        yt = hcat(y1_rev, y2[:, 2:end])
        zt = hcat(z1_rev, z2[:, 2:end])
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

function trace3d_rk4(
        fieldx, fieldy, fieldz, startx::AbstractVector, starty::AbstractVector, startz::AbstractVector,
        gridx, gridy, gridz; maxstep::Int = 20000, ds::T = 0.01, gridtype::String = "ndgrid",
        direction::String = "both"
    ) where {T}

    if gridtype == "ndgrid"
        fx, fy, fz = fieldx, fieldy, fieldz
    else # meshgrid
        fx, fy, fz = permutedims(fieldx), permutedims(fieldy), permutedims(fieldz)
    end

    if direction == "forward"
        xt, yt, zt, _ = rk4_batch(maxstep, ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    elseif direction == "backward"
        xt, yt, zt, _ =
            rk4_batch(maxstep, -ds, startx, starty, startz, gridx, gridy, gridz, fx, fy, fz)
    else
        x1, y1, z1, _ = rk4_batch(
            floor(Int, maxstep / 2), -ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )
        x2, y2, z2, _ = rk4_batch(
            maxstep - size(x1, 2) + 1, ds, startx, starty, startz,
            gridx, gridy, gridz, fx, fy, fz
        )

        x1_rev = reverse(x1, dims = 2)
        y1_rev = reverse(y1, dims = 2)
        z1_rev = reverse(z1, dims = 2)

        xt = hcat(x1_rev, x2[:, 2:end])
        yt = hcat(y1_rev, y2[:, 2:end])
        zt = hcat(z1_rev, z2[:, 2:end])
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
