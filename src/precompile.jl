# Precompiling workloads

@setup_workload begin
    @compile_workload begin
        # 2D Data
        x = range(0, 1, step = 0.1)
        y = range(0, 1, step = 0.1)
        u = fill(1.0, length(x), length(y))
        v = fill(1.0, length(x), length(y))

        # Single particle
        xstart, ystart = 0.5, 0.5
        trace(u, v, xstart, ystart, x, y; maxstep = 10, ds = 0.1, alg = Euler())
        trace(u, v, xstart, ystart, x, y; maxstep = 10, ds = 0.1, alg = RK4())

        # Batch particles (Vector input)
        xstarts = [0.1, 0.5, 0.9]
        ystarts = [0.1, 0.5, 0.9]
        trace(u, v, xstarts, ystarts, x, y; maxstep = 10, ds = 0.1, alg = Euler())
        trace(u, v, xstarts, ystarts, x, y; maxstep = 10, ds = 0.1, alg = RK4())

        # 3D Data
        x3 = range(0, 1, length = 5)
        y3 = range(0, 1, length = 5)
        z3 = range(0, 1, length = 5)
        bx = fill(1.0, 5, 5, 5)
        by = fill(1.0, 5, 5, 5)
        bz = fill(1.0, 5, 5, 5)

        # Single particle
        xs, ys, zs = 0.5, 0.5, 0.5
        trace(bx, by, bz, xs, ys, zs, x3, y3, z3; maxstep = 5, ds = 0.1, alg = Euler())
        trace(bx, by, bz, xs, ys, zs, x3, y3, z3; maxstep = 5, ds = 0.1, alg = RK4())

        # Batch particles
        xstarts3 = [0.1, 0.5]
        ystarts3 = [0.1, 0.5]
        zstarts3 = [0.1, 0.5]
        trace(bx, by, bz, xstarts3, ystarts3, zstarts3, x3, y3, z3; maxstep = 5, ds = 0.1, alg = Euler())
        trace(bx, by, bz, xstarts3, ystarts3, zstarts3, x3, y3, z3; maxstep = 5, ds = 0.1, alg = RK4())
    end
end
