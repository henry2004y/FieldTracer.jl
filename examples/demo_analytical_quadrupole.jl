using FieldTracer
using LinearAlgebra: norm

"""
    quadrupole(u, s)

Axisymmetric magnetic quadrupole aligned with z, in dimensionless units.
Outside the origin, B = -grad(Phi), where Phi = (3z^2 - r^2) / (2r^5).
"""
function quadrupole(u, s)
    x, y, z = u
    r2 = x^2 + y^2 + z^2
    iszero(r2) && throw(DomainError(u, "The quadrupole is singular at the origin."))
    scale = 1.5 / r2^(7 / 2)
    return (
        scale * x * (5z^2 - r2),
        scale * y * (5z^2 - r2),
        scale * z * (5z^2 - 3r2),
    )
end

"""
    trace_quadrupole(; ds=0.01)

Trace seeds on a radius-2 circle in the x-z plane, excluding a radius-0.5
source region and stopping outside radius 4. Both directions are traced.
"""
function trace_quadrupole(; ds = 0.01)
    angles = range(0, 2pi; length = 25)[1:end-1] .+ pi / 24
    seeds = [[2sin(theta), 0.0, 2cos(theta)] for theta in angles]
    isoutside(u, s) = !(0.5 < norm(u) < 4.0)
    traces = [
        trace(quadrupole, seed; ds, max_length = 20.0, maxstep = 10000, isoutside)
        for seed in seeds
    ]
    return (; seeds, traces)
end

# Plotting is optional: including this file only defines the model and tracing helper.
if abspath(PROGRAM_FILE) == @__FILE__
    using PyPlot

    result = trace_quadrupole()
    fig, ax = subplots(; figsize = (7, 7))
    for line in result.traces
        for branch in (line.forward, line.backward)
            ax.plot(
                [u[1] for u in branch.points], [u[3] for u in branch.points];
                color = "C0", linewidth = 1,
            )
        end
    end
    theta = range(0, 2pi; length = 201)
    ax.fill(0.5 .* cos.(theta), 0.5 .* sin.(theta); color = "0.8")
    ax.scatter(
        [u[1] for u in result.seeds], [u[3] for u in result.seeds];
        color = "C1", s = 12, label = "Seeds",
    )
    ax.set(; xlabel = "x", ylabel = "z", title = "Analytical quadrupole field",
        xlim = (-4, 4), ylim = (-4, 4), aspect = "equal")
    ax.legend()
    fig.tight_layout()
    show()
end
