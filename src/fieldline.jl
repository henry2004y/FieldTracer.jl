"""
    trace_fieldline!(du, u, field, s)

ODE for tracing a magnetic field line parameterized by arc length. `field(u, s)`
must return the magnetic field vector at position `u` and arc length `s`.
"""
function trace_fieldline!(du, u, field, s)
    B = field(u, s)
    length(du) == length(B) ||
        throw(DimensionMismatch("field and state dimensions must match"))

    Bmag = norm(B)
    if iszero(Bmag)
        fill!(du, zero(eltype(du)))
    else
        @inbounds for i in eachindex(du, B)
            du[i] = B[i] / Bmag
        end
    end
    return nothing
end

"""
    TraceFieldlineProblem(u0, field, tspan; mode=:both)

Create SciML `ODEProblem`s for tracing a magnetic field line through the analytical
field function `field(u, s)`. The independent variable `s` is arc length. `mode`
may be `:forward`, `:backward`, or `:both`; the latter returns a pair of problems.
Use `solve` with an ODE solver such as those provided by OrdinaryDiffEq.jl.
"""
function TraceFieldlineProblem(
        u0, field, tspan::Tuple{<:Real, <:Real}; mode::Symbol = :both
    )
    tstart, tend = tspan
    if mode === :forward
        return ODEProblem(trace_fieldline!, u0, tspan, field)
    elseif mode === :backward
        backward_tspan = (tstart, tstart - abs(tend - tstart))
        return ODEProblem(trace_fieldline!, u0, backward_tspan, field)
    elseif mode === :both
        forward = ODEProblem(trace_fieldline!, u0, tspan, field)
        backward_tspan = (tstart, tstart - abs(tend - tstart))
        backward = ODEProblem(trace_fieldline!, u0, backward_tspan, field)
        return forward, backward
    else
        throw(ArgumentError("Unsupported mode: $mode. Expected :forward, :backward, or :both."))
    end
end
