```@meta
CurrentModule = FieldTracer
```

# FieldTracer

Vector field tracing on common types of meshes and in analytical fields.

This package supports 2nd and 4th order field line tracing on
* 2D/3D regular Cartesian mesh
* 2D unstructured quadrilateral mesh
* 2D unstructured triangular mesh

It can be used for
* generating streamlines and fieldlines;
* checking field connectivity between spatial cells.

## Analytical fields

Use [`trace`](@ref) to trace an analytical field with native fixed-step RK4.
No ODE solver package is required. The field function takes a position and signed
arc length from the seed, and returns the field vector:

```julia
using FieldTracer

B(x, s) = (0.0, 0.0, 1.0)
result = trace(B, [0.0, 0.0, 0.0]; ds = 0.1, max_length = 10.0)
result.forward.points[end]  # approximately [0.0, 0.0, 10.0]
result.backward.points[end] # approximately [0.0, 0.0, -10.0]
result.forward.status      # :max_length
```

`direction` accepts `"forward"`, `"backward"`, or `"both"` (the default).
A single direction returns `(points, s, status)`; both directions return named
`forward` and `backward` results, each beginning at the seed. `points` contains
position vectors and `s` contains signed arc lengths.

`ds` is a positive physical arc-length step. Each branch saves at most `maxstep`
points (default 20000, including the seed), and traces at most `max_length`
(default `Inf`). A final partial step reaches a finite length limit.

Pass `isoutside = (x, s) -> ...` to restrict the domain. The predicate is checked
before every field evaluation, including intermediate RK stages, and before
accepting an endpoint. If any stage exits, the entire step is discarded; no
boundary intersection is computed.

Termination is reported as `:max_length`, `:maxstep`, `:outside`, `:zero_field`,
`:nonfinite_field`, `:nonfinite_state`, or `:stepsize_underflow`. Exactly zero
fields stop tracing; nonzero fields, including near-zero fields, are normalized.
Exceptions from your field or domain predicate are not suppressed.

Fixed-step RK4 has no automatic error control. Check convergence with `ds/2`,
particularly near magnetic nulls or regions of strong curvature. Use an adaptive
solver when tolerance control or accurate boundary-event location is needed.

### Optional SciML integration

Loading SciMLBase enables [`TraceFieldlineProblem`](@ref) through a package
extension. To integrate adaptively, install and load the focused
[OrdinaryDiffEqTsit5.jl](https://docs.sciml.ai/OrdinaryDiffEq/stable/explicit/Tsit5/)
solver package, which also loads SciMLBase; the full OrdinaryDiffEq package is
not required:

```julia
using FieldTracer, OrdinaryDiffEqTsit5

B(x, s) = (0.0, 0.0, 1.0)
prob = TraceFieldlineProblem([0.0, 0.0, 0.0], B, (0.0, 10.0); mode = :forward)
solution = solve(prob, Tsit5())
```

Existing `TraceFieldlineProblem` calls retain their behavior, but now require
loading SciMLBase or a solver package explicitly.