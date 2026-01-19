# Test functions for tracing in different kinds of analytical fields.

include("utility/dipole.jl")

"""
	 test_trace_asymptote(issingle=false)

Test streamline tracing by plotting vectors and associated streamlines through a
simple velocity field where Vx=x, Vy=-y. Support for single and double precision
data.
"""
function test_trace_asymptote(issingle = false)
    # Start by creating a velocity vector field.
    if issingle
        xmax, ymax = 200.0f0, 20.0f0
        x = -10.0f0:0.25f0:xmax
        y = -10.0f0:0.25f0:ymax
    else
        xmax, ymax = 200.0, 20.0
        x = -10.0:0.25:xmax
        y = -10.0:0.25:ymax
    end

    xgrid = [i for j in y, i in x]
    ygrid = [j for j in y, i in x]

    if issingle
        vx = xgrid * 1.0f0
        vy = ygrid * -1.0f0

        xstart = 1.0f0
        ystart = 10.0f0
    else
        vx = xgrid * 1.0
        vy = ygrid * -1.0

        xstart = 1.0
        ystart = 10.0
    end

    ds = 0.1
    gridtype = "meshgrid"

    x1, y1 = trace(vx, vy, xstart, ystart, x, y; alg = RK4(), ds = 0.025, gridtype)
    x2, y2 = trace(vx, vy, xstart, ystart, x, y; alg = Euler(), ds = 0.025, gridtype)

    # analytical solution const = x*y
    c = xstart * ystart

    return if issingle
        if isapprox(x1[end] * y1[end], c, atol = 1.0e-2) &&
                isapprox(x2[end] * y2[end], c, atol = 0.13)
            return true
        else
            return false
        end
    else
        if isapprox(x1[end] * y1[end], c, atol = 1.0e-4) &&
                isapprox(x2[end] * y2[end], c, atol = 0.13)
            return true
        else
            return false
        end
    end
end

"Trace field lines through a Earth like magnetic dipole field."
function test_trace_dipole()

    # Start by creating a field of unit vectors.
    x = -100.0:5.0:101.0
    y = -100.0:5.0:101.0

    bx, by = b_hat(x, y)

    # Trace through this field.
    xstart = 10.0
    ystart = 25.0
    gridtype = "meshgrid"

    x1, y1 = trace(bx, by, xstart, ystart, x, y; alg = RK4(), ds = 0.5, gridtype)
    x2, y2 = trace(bx, by, xstart, ystart, x, y; alg = Euler(), ds = 0.5, gridtype)

    if length(x1) == 258 && isapprox(x1[end] * y1[end], 7476.155, atol = 0.1) &&
            length(x2) == 260 && isapprox(x2[end] * y2[end], 7574.73, atol = 0.1)
        return true
    else
        return false
    end
end

"Test tracing on an anisotropic grid to verify physical stepping."
function test_trace_anisotropy()
    # Anisotropic grid
    x = -200.0:100.0:200.0 # dx = 100
    y = -2.0:1.0:2.0       # dy = 1

    # Field 1: (1, 0)
    fx1 = ones(length(y), length(x))
    fy1 = zeros(length(y), length(x))

    startx, starty = 0.0, 0.0
    ds = 0.01 # Physical step size

    # Trace X
    x1, y1 = trace(fx1, fy1, startx, starty, x, y; alg = Euler(), ds = ds, maxstep = 5)

    dx_phys = sqrt((x1[2] - x1[1])^2 + (y1[2] - y1[1])^2)

    # Field 2: (0, 1)
    fx2 = zeros(length(y), length(x))
    fy2 = ones(length(y), length(x))

    # Trace Y
    x2, y2 = trace(fx2, fy2, startx, starty, x, y; alg = Euler(), ds = ds, maxstep = 5)

    dy_phys = sqrt((x2[2] - x2[1])^2 + (y2[2] - y2[1])^2)

    # Check if physical step is approximately ds and isotropic
    return isapprox(dx_phys, ds, atol = 1.0e-5) &&
        isapprox(dy_phys, ds, atol = 1.0e-5) &&
        isapprox(dx_phys, dy_phys, atol = 1.0e-5)
end
