# FieldTracer

Vector field tracer on common types of meshes and in analytical fields.

[![Build Status](https://github.com/henry2004y/FieldTracer.jl/workflows/CI/badge.svg)](https://github.com/henry2004y/FieldTracer.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://henry2004y.github.io/FieldTracer.jl/dev)
[![Coverage](https://codecov.io/gh/henry2004y/FieldTracer.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/henry2004y/FieldTracer.jl)

Analytical fields can be traced with native fixed-step RK4, without an ODE solver:

```julia
using FieldTracer

B(x, s) = (0.0, 0.0, 1.0)
result = trace(B, [0.0, 0.0, 0.0]; ds = 0.1, max_length = 10.0)
result.forward.points
result.forward.status # :max_length
```

For adaptive integration, load SciMLBase or OrdinaryDiffEqTsit5 to enable the
optional `TraceFieldlineProblem` interface. See the
[documentation](https://henry2004y.github.io/FieldTracer.jl/dev) for details.
