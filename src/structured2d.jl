# 2D Field tracing on a regular grid.

"""
    bilin_reg(x, y, Q00, Q01, Q10, Q11)

Bilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1)
Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.
"""
@inline function bilin_reg(x, y, Q00, Q10, Q01, Q11)
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
@inline grid_interp(x, y, ix, iy, field::Array) =
    bilin_reg(
    x - ix, y - iy,
    field[ix + 1, iy + 1],
    field[ix + 2, iy + 1],
    field[ix + 1, iy + 2],
    field[ix + 2, iy + 2]
)

@inline check_domain(ix, iy, iSize, jSize) =
    (ix >= 0) & (ix < iSize - 1) & (iy >= 0) & (iy < jSize - 1)

@inline check_validity(fx, fy, v_mag) =
    (v_mag > 0) & !isnan(fx) & !isnan(fy) & !isinf(fx) & !isinf(fy)

@inline Base.@propagate_inbounds function _euler_step(cx, cy, ux, uy, ds, inv_dx, inv_dy, iSize, jSize)
    ix = floor(Int, cx)
    iy = floor(Int, cy)

    # Check domain
    in_domain = check_domain(ix, iy, iSize, jSize)

    # Clamp indices for safe memory access even if out of domain
    ix_c = max(0, min(ix, iSize - 2))
    iy_c = max(0, min(iy, jSize - 2))

    # Interpolate
    fx = grid_interp(cx, cy, ix_c, iy_c, ux)
    fy = grid_interp(cx, cy, ix_c, iy_c, uy)

    v_mag = hypot(fx, fy)

    # Check for validity
    valid = in_domain & check_validity(fx, fy, v_mag)

    v_inv = ifelse(valid, inv(v_mag), 0.0)

    # Step size in index space
    dx = ifelse(valid, ds * fx * v_inv * inv_dx, 0.0)
    dy = ifelse(valid, ds * fy * v_inv * inv_dy, 0.0)

    return cx + dx, cy + dy, valid
end

@inline Base.@propagate_inbounds function _rk4_step(cx, cy, ux, uy, ds, inv_dx, inv_dy, iSize, jSize)
    # --- Substep 1 ---
    ix = floor(Int, cx)
    iy = floor(Int, cy)
    valid = check_domain(ix, iy, iSize, jSize)
    ix_c, iy_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2))

    f1x = grid_interp(cx, cy, ix_c, iy_c, ux)
    f1y = grid_interp(cx, cy, ix_c, iy_c, uy)
    v1_mag = hypot(f1x, f1y)

    valid &= check_validity(f1x, f1y, v1_mag)
    v1_inv = ifelse(valid, inv(v1_mag), 0.0)

    k1x = ifelse(valid, f1x * v1_inv * inv_dx, 0.0)
    k1y = ifelse(valid, f1y * v1_inv * inv_dy, 0.0)

    # --- Substep 2 ---
    px = cx + k1x * ds * 0.5
    py = cy + k1y * ds * 0.5

    ix = floor(Int, px)
    iy = floor(Int, py)
    valid &= check_domain(ix, iy, iSize, jSize)
    ix_c, iy_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2))

    f2x = grid_interp(px, py, ix_c, iy_c, ux)
    f2y = grid_interp(px, py, ix_c, iy_c, uy)
    v2_mag = hypot(f2x, f2y)

    valid &= check_validity(f2x, f2y, v2_mag)
    v2_inv = ifelse(valid, inv(v2_mag), 0.0)

    k2x = ifelse(valid, f2x * v2_inv * inv_dx, 0.0)
    k2y = ifelse(valid, f2y * v2_inv * inv_dy, 0.0)

    # --- Substep 3 ---
    px = cx + k2x * ds * 0.5
    py = cy + k2y * ds * 0.5

    ix = floor(Int, px)
    iy = floor(Int, py)
    valid &= check_domain(ix, iy, iSize, jSize)
    ix_c, iy_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2))

    f3x = grid_interp(px, py, ix_c, iy_c, ux)
    f3y = grid_interp(px, py, ix_c, iy_c, uy)
    v3_mag = hypot(f3x, f3y)

    valid &= check_validity(f3x, f3y, v3_mag)
    v3_inv = ifelse(valid, inv(v3_mag), 0.0)

    k3x = ifelse(valid, f3x * v3_inv * inv_dx, 0.0)
    k3y = ifelse(valid, f3y * v3_inv * inv_dy, 0.0)

    # --- Substep 4 ---
    px = cx + k3x * ds
    py = cy + k3y * ds

    ix = floor(Int, px)
    iy = floor(Int, py)
    valid &= check_domain(ix, iy, iSize, jSize)
    ix_c, iy_c = max(0, min(ix, iSize - 2)), max(0, min(iy, jSize - 2))

    f4x = grid_interp(px, py, ix_c, iy_c, ux)
    f4y = grid_interp(px, py, ix_c, iy_c, uy)
    v4_mag = hypot(f4x, f4y)

    valid &= check_validity(f4x, f4y, v4_mag)
    v4_inv = ifelse(valid, inv(v4_mag), 0.0)

    k4x = ifelse(valid, f4x * v4_inv * inv_dx, 0.0)
    k4y = ifelse(valid, f4y * v4_inv * inv_dy, 0.0)

    # Update
    next_x = cx + ds / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    next_y = cy + ds / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)

    return next_x, next_y, valid
end

"""
    euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast 2D tracing using Euler's method. It takes at most `maxstep` with step size `ds`
tracing the vector field given by `ux,uy` starting from `(startx,starty)` in the Cartesian
grid specified by ranges `xGrid` and `yGrid`. Step size is in physical unit.
Return footprints' coordinates in (`x`, `y`).
"""
@muladd function euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
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

    nstep = 1
    # Perform tracing using Euler's method
    @inbounds @fastmath for n in 1:(maxstep - 1)
        next_x, next_y, valid = _euler_step(x[n], y[n], ux, uy, ds, inv_dx, inv_dy, iSize, jSize)

        if !valid
            break
        end

        x[n + 1] = next_x
        y[n + 1] = next_y
        nstep = n + 1
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        x[i] = x[i] * dx + xGrid[1]
        y[i] = y[i] * dy + yGrid[1]
    end

    return x[1:nstep], y[1:nstep]
end

"""
    rk4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and constant step
size `ds`. See also [`euler`](@ref).
"""
@muladd function rk4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
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

    nstep = 1
    # Perform tracing using RK4's method
    @inbounds @fastmath for n in 1:(maxstep - 1)
        next_x, next_y, valid = _rk4_step(x[n], y[n], ux, uy, ds, inv_dx, inv_dy, iSize, jSize)

        if !valid
            break
        end

        x[n + 1] = next_x
        y[n + 1] = next_y
        nstep = n + 1
    end

    # Convert traced points to original coordinate system.
    @inbounds @simd for i in 1:nstep
        x[i] = x[i] * dx + xGrid[1]
        y[i] = y[i] * dy + yGrid[1]
    end

    return x[1:nstep], y[1:nstep]
end

"""
    euler_batch(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast 2D tracing of multiple particles using Euler's method with LoopVectorization.
Returns matrices `x` and `y` of size `(n_particles, maxstep)`.
"""
function euler_batch(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
    n_particles = length(startx)
    @assert length(starty) == n_particles "startx and starty must have same length"
    @assert size(ux) == size(uy) "field array sizes must be equal!"

    x = Matrix{eltype(startx)}(undef, n_particles, maxstep)
    y = Matrix{eltype(starty)}(undef, n_particles, maxstep)

    iSize, jSize = size(xGrid, 1), size(yGrid, 1)

    # Get starting points in normalized/array coordinates
    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    inv_dx, inv_dy = inv(dx), inv(dy)

    x0, y0 = xGrid[1], yGrid[1]

    @inbounds @fastmath @simd for i in 1:n_particles
        x[i, 1] = (startx[i] - x0) * inv_dx
        y[i, 1] = (starty[i] - y0) * inv_dy
    end

    # Perform tracing using Euler's method
    # Time loop is outer, particle loop is inner (vectorized)
    nsteps = ones(Int, n_particles)

    for n in 1:(maxstep - 1)
        active_count = 0
        @inbounds @fastmath @simd for i in 1:n_particles
            # Current normalized position
            cx = x[i, n]
            cy = y[i, n]

            next_x, next_y, valid = _euler_step(cx, cy, ux, uy, ds, inv_dx, inv_dy, iSize, jSize)

            # Update active count and nsteps
            # valid is boolean, promotes to 0/1 integer in addition
            active_count += valid
            nsteps[i] += valid

            x[i, n + 1] = next_x
            y[i, n + 1] = next_y
        end

        if active_count == 0
            break
        end
    end

    # Convert traces back to physical coordinates
    @inbounds @fastmath @simd for n in 1:maxstep
        for i in 1:n_particles
            x[i, n] = x[i, n] * dx + x0
            y[i, n] = y[i, n] * dy + y0
        end
    end

    return x, y, nsteps
end

"""
    rk4_batch(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)

Fast 2D tracing of multiple particles using RK4 method with LoopVectorization.
Returns matrices `x` and `y` of size `(n_particles, maxstep)`.
"""
function rk4_batch(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)
    n_particles = length(startx)
    @assert length(starty) == n_particles "startx and starty must have same length"
    @assert size(ux) == size(uy) "field array sizes must be equal!"

    x = Matrix{eltype(startx)}(undef, n_particles, maxstep)
    y = Matrix{eltype(starty)}(undef, n_particles, maxstep)

    iSize, jSize = size(xGrid, 1), size(yGrid, 1)

    dx = xGrid[2] - xGrid[1]
    dy = yGrid[2] - yGrid[1]
    inv_dx, inv_dy = inv(dx), inv(dy)
    x0, y0 = xGrid[1], yGrid[1]

    @inbounds @fastmath @simd for i in 1:n_particles
        x[i, 1] = (startx[i] - x0) * inv_dx
        y[i, 1] = (starty[i] - y0) * inv_dy
    end

    nsteps = ones(Int, n_particles)

    for n in 1:(maxstep - 1)
        active_count = 0
        @inbounds @fastmath @simd for i in 1:n_particles
            cx = x[i, n]
            cy = y[i, n]

            next_x, next_y, valid = _rk4_step(cx, cy, ux, uy, ds, inv_dx, inv_dy, iSize, jSize)

            # Update active count and nsteps
            active_count += valid
            nsteps[i] += valid

            x[i, n + 1] = next_x
            y[i, n + 1] = next_y
        end

        if active_count == 0
            break
        end
    end

    @inbounds @fastmath @simd for n in 1:maxstep
        for i in 1:n_particles
            x[i, n] = x[i, n] * dx + x0
            y[i, n] = y[i, n] * dy + y0
        end
    end

    return x, y, nsteps
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

function trace2d_rk4(
        fieldx, fieldy, startx::AbstractVector, starty::AbstractVector, gridx, gridy;
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
        # nsteps is ignored for batch scalar tracing
        xt, yt, _ = rk4_batch(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
    elseif direction == "backward"
        xt, yt, _ = rk4_batch(maxstep, -ds, startx, starty, gridx, gridy, fx, fy)
    else
        x1, y1, _ = rk4_batch(floor(Int, maxstep / 2), -ds, startx, starty, gridx, gridy, fx, fy)
        x2, y2, _ = rk4_batch(maxstep - size(x1, 2) + 1, ds, startx, starty, gridx, gridy, fx, fy)

        # Combine traces (concatenate along dim 2, reverse x1 first)
        # Note: batch returns full matrices, so we might have some zero padding if stopped early?
        # Actually our batch implementation returns full maxstep size.
        # But for bidirectional, it's tricky since batch tracing typically expects uniform steps.
        # For simplicity, we just concat.

        # Reverse x1 columns
        x1_rev = reverse(x1, dims = 2)
        y1_rev = reverse(y1, dims = 2)

        # Concat, skipping the first point of x2 (which duplicates last of x1 inverted)
        xt = hcat(x1_rev, x2[:, 2:end])
        yt = hcat(y1_rev, y2[:, 2:end])
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
        # nsteps is ignored, we rely on checking constants or trusting the user knows batch size
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

function trace2d_euler(
        fieldx, fieldy, startx::AbstractVector, starty::AbstractVector, gridx, gridy;
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
        xt, yt, _ = euler_batch(maxstep, ds, startx, starty, gridx, gridy, fx, fy)
    elseif direction == "backward"
        xt, yt, _ = euler_batch(maxstep, -ds, startx, starty, gridx, gridy, fx, fy)
    else
        x1, y1, _ = euler_batch(floor(Int, maxstep / 2), -ds, startx, starty, gridx, gridy, fx, fy)
        x2, y2, _ = euler_batch(maxstep - size(x1, 2) + 1, ds, startx, starty, gridx, gridy, fx, fy)

        x1_rev = reverse(x1, dims = 2)
        y1_rev = reverse(y1, dims = 2)

        xt = hcat(x1_rev, x2[:, 2:end])
        yt = hcat(y1_rev, y2[:, 2:end])
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
