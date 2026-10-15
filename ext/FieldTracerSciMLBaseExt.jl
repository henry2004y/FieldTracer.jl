module FieldTracerSciMLBaseExt

using FieldTracer: FieldTracer, trace_fieldline!
using SciMLBase: ODEProblem

function FieldTracer.TraceFieldlineProblem(
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

end
