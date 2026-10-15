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
        @inbounds for (i, b) in zip(eachindex(du), B)
            du[i] = b / Bmag
        end
    end
    return nothing
end

"""
    TraceFieldlineProblem(u0, field, tspan; mode=:both)

Create SciML `ODEProblem`s for tracing a magnetic field line through the analytical
field function `field(u, s)`. The independent variable `s` is arc length. `mode`
may be `:forward`, `:backward`, or `:both`; the latter returns a pair of problems.
Load SciMLBase (or OrdinaryDiffEqTsit5) to enable this optional interface, then
use `solve` with a solver such as `Tsit5()`. For native fixed-step tracing, use
[`trace`](@ref) instead.
"""
function TraceFieldlineProblem end

"""
    trace(field, u0::AbstractVector; alg=RK4(), ds=0.01, maxstep=20000,
          max_length=Inf, direction="both", isoutside=(u, s) -> false)

Trace the analytical vector field `field(u, s)` with native fixed-step RK4.
`u0` is a nonempty real position vector and `s` is signed arc length from the
seed. `ds` must be positive. `direction` is `"forward"`, `"backward"`, or `"both"`.
Only `RK4()` is supported for analytical fields.

Each direction returns a named tuple `(points, s, status)`, where `points` is a
vector of position vectors and `s` contains their signed arc lengths. With
`direction="both"`, return `(forward=..., backward=...)`; each branch starts at
the seed and has its own limits. `maxstep` limits the number of saved points
(including the seed). `max_length` limits arc length per branch; the last step
is shortened to reach it.

`isoutside(u, s)` is checked before every field evaluation and before saving an
endpoint. An invalid intermediate RK stage discards the entire step. No boundary
intersection is computed; the result retains the last accepted point.

`status` is `:max_length`, `:maxstep`, `:outside`, `:zero_field`,
`:nonfinite_field`, `:nonfinite_state`, or `:stepsize_underflow`.
Exactly zero fields terminate tracing; near-zero nonzero fields are normalized.
Exceptions from the field or predicate propagate to the caller.

There is no adaptive error control. Check convergence by reducing `ds`, or use
[`TraceFieldlineProblem`](@ref) with an optional adaptive SciML solver.
"""
function trace(
        field, u0::AbstractVector{<:Real}; alg::Algorithm = RK4(),
        ds::Real = 0.01, maxstep::Integer = 20000, max_length::Real = Inf,
        direction = "both", isoutside = (u, s) -> false
    )
    alg isa RK4 || throw(ArgumentError("Analytical tracing supports only RK4()."))
    isempty(u0) && throw(ArgumentError("u0 must be nonempty."))
    all(isfinite, u0) || throw(ArgumentError("u0 must be finite."))
    isfinite(ds) && ds > 0 || throw(ArgumentError("ds must be finite and positive."))
    maxstep >= 1 || throw(ArgumentError("maxstep must be at least 1."))
    max_length >= 0 || throw(ArgumentError("max_length must be nonnegative."))
    direction in ("forward", "backward", "both") ||
        throw(ArgumentError("direction must be \"forward\", \"backward\", or \"both\"."))

    T = promote_type(typeof(float(ds)), mapreduce(x -> typeof(float(x)), promote_type, u0))
    seed = T[x for x in u0]
    step = T(ds)
    limit = T(max_length)
    isfinite(step) && step > 0 ||
        throw(ArgumentError("ds is not representable as a finite positive step."))
    isfinite(max_length) && !isfinite(limit) &&
        throw(ArgumentError("max_length is not representable in the state precision."))
    max_length > 0 && iszero(limit) &&
        throw(ArgumentError("max_length underflows in the state precision."))
    if direction == "both"
        forward = _trace_analytical(field, seed, step, maxstep, limit, isoutside)
        backward = _trace_analytical(field, seed, -step, maxstep, limit, isoutside)
        return (; forward, backward)
    else
        h = direction == "forward" ? step : -step
        return _trace_analytical(field, seed, h, maxstep, limit, isoutside)
    end
end

function _analytical_direction!(du, field, u, s, isoutside)
    all(isfinite, u) || return :nonfinite_state
    isoutside(u, s) && return :outside
    B = field(u, s)
    length(du) == length(B) ||
        throw(DimensionMismatch("field and state dimensions must match"))
    all(x -> x isa Real, B) || throw(ArgumentError("field values must be real."))
    all(isfinite, B) || return :nonfinite_field
    magnitude = norm(B)
    isfinite(magnitude) || return :nonfinite_field
    iszero(magnitude) && return :zero_field
    for (i, b) in enumerate(B)
        du[i] = b / magnitude
    end
    return :continue
end

function _trace_analytical(
        field::F, seed::Vector{T}, ds, maxstep, limit, isoutside::P
    ) where {F, T, P}
    u = copy(seed)
    points = [copy(u)]
    svalues = T[0]
    k1, k2, k3, k4, stage = (similar(u) for _ in 1:5)
    s = zero(T)
    status = _analytical_direction!(k1, field, u, s, isoutside)
    while status === :continue
        if abs(s) >= limit
            status = :max_length
            break
        elseif length(points) >= maxstep
            status = :maxstep
            break
        end

        remaining = limit - abs(s)
        last_step = remaining <= abs(ds)
        h = copysign(min(abs(ds), remaining), ds)
        snext = last_step ? copysign(limit, ds) : s + h
        if !isfinite(snext) || snext == s || s + h / 2 == s
            status = :stepsize_underflow
            break
        end

        @. stage = u + h / 2 * k1
        status = _analytical_direction!(k2, field, stage, s + h / 2, isoutside)
        status === :continue || break
        @. stage = u + h / 2 * k2
        status = _analytical_direction!(k3, field, stage, s + h / 2, isoutside)
        status === :continue || break
        @. stage = u + h * k3
        status = _analytical_direction!(k4, field, stage, snext, isoutside)
        status === :continue || break

        @. stage = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if !all(isfinite, stage)
            status = :nonfinite_state
            break
        elseif isoutside(stage, snext)
            status = :outside
            break
        end
        copyto!(u, stage)
        s = snext
        push!(points, copy(u))
        push!(svalues, s)
        if last_step
            status = :max_length
        elseif length(points) >= maxstep
            status = :maxstep
        else
            status = _analytical_direction!(k1, field, u, s, isoutside)
        end
    end
    return (; points, s = svalues, status)
end
