using FieldTracer
using Test
using LinearAlgebra: norm

@testset "Native analytical tracing" begin
    straight(u, s) = (0.0, 0.0, 2.0)
    seed = [1.0, 2.0, 3.0]
    result = trace(straight, seed; direction = "forward", ds = 0.3, max_length = 1.0)
    @test keys(result) == (:points, :s, :status)
    @test result.status === :max_length
    @test result.s ≈ [0.0, 0.3, 0.6, 0.9, 1.0]
    @test result.points[end] ≈ [1.0, 2.0, 4.0]
    @test seed == [1.0, 2.0, 3.0]
    @test result.points[1] !== seed
    @test result.points[1] !== result.points[2]

    both = trace(straight, seed; ds = 0.3, max_length = 1.0)
    @test keys(both) == (:forward, :backward)
    @test both.forward.points == result.points
    @test both.backward.points[end] ≈ [1.0, 2.0, 2.0]
    @test both.backward.s ≈ -result.s
    @test both.backward.status === :max_length
    backward = trace(straight, seed; direction = "backward", ds = 0.3, max_length = 1.0)
    @test backward == both.backward

    limited = trace(straight, seed; direction = "forward", maxstep = 3, ds = 0.2)
    @test limited.status === :maxstep
    @test length(limited.points) == length(limited.s) == 3
    @test limited.s[end] ≈ 0.4
    @test trace(straight, seed; direction = "forward", maxstep = 1).points == [seed]
    zero_length = trace(straight, seed; direction = "forward", max_length = 0)
    @test zero_length.status === :max_length
    @test zero_length.s == [0.0]

    single = trace((u, s) -> (1.0f0, 0.0f0), Float32[0, 0];
        ds = 0.1f0, max_length = 0.25f0, direction = "forward")
    @test eltype(single.s) === Float32
    @test eltype(single.points[end]) === Float32
    @test single.points[end] ≈ Float32[0.25, 0]
    integers = trace((u, s) -> (1, 0), [0, 0]; maxstep = 2, direction = "forward")
    @test integers.points[end] ≈ [0.01, 0]
    callable = trace(Returns([1.0, 0.0]), [0.0, 0.0];
        direction = "forward", maxstep = 2, isoutside = Returns(false))
    @test callable.points[end] ≈ [0.01, 0]

    @testset "Convergence and geometry" begin
        circle(u, s) = (-u[2], u[1])
        exact = [cos(2.0), sin(2.0)]
        errors = map((0.2, 0.1, 0.05)) do ds
            sol = trace(circle, [1.0, 0.0]; ds, max_length = 2.0, direction = "forward")
            @test sol.status === :max_length
            norm(sol.points[end] - exact)
        end
        @test 12 < errors[1] / errors[2] < 20
        @test 12 < errors[2] / errors[3] < 20

        dipole(u, s) = (3 * u[1] * u[3], 3 * u[2] * u[3],
            3 * u[3]^2 - sum(abs2, u))
        sol = trace(dipole, [2.0, 0.0, 0.0]; ds = 0.01,
            max_length = 2.0, direction = "forward")
        @test sol.status === :max_length
        @test all(sol.points) do u
            r = norm(u)
            isapprox(r^3, 2 * (u[1]^2 + u[2]^2); atol = 1.0e-8)
        end

        # For f(s) = (cos(s), sin(s)), the exact integral is known in both directions.
        nonautonomous(u, s) = (cos(s), sin(s))
        sol = trace(nonautonomous, [0.0, 0.0]; ds = 0.01, max_length = 1.0)
        @test sol.forward.points[end] ≈ [sin(1.0), 1 - cos(1.0)] atol = 1.0e-10
        @test sol.backward.points[end] ≈ [-sin(1.0), 1 - cos(1.0)] atol = 1.0e-10
    end

    @testset "Termination and invalid stages" begin
        outside(u, s) = u[1] > 0.75
        guarded(u, s) = outside(u, s) ? error("field evaluated outside domain") : (1.0, 0.0)
        sol = trace(guarded, [0.0, 0.0]; ds = 1.0,
            direction = "forward", isoutside = outside)
        @test sol.status === :outside
        @test sol.points == [[0.0, 0.0]]
        @test trace(guarded, [1.0, 0.0]; isoutside = outside,
            direction = "forward").status === :outside
        sol = trace(guarded, [0.0, 0.0]; ds = 0.5,
            direction = "forward", isoutside = outside)
        @test sol.points[end] ≈ [0.5, 0]

        endpoint_field(u, s) = s == 0 || s == 1 ? (1.0, 0.0) : (0.0, 1.0)
        endpoint_outside(u, s) = u[1] > 0.25 && u[2] > 0.6
        sol = trace(endpoint_field, [0.0, 0.0]; ds = 1.0,
            direction = "forward", isoutside = endpoint_outside)
        @test sol.status === :outside
        @test sol.points == [[0.0, 0.0]]

        for (field, status) in (
                ((u, s) -> (0.0, 0.0), :zero_field),
                ((u, s) -> (NaN, 0.0), :nonfinite_field),
                ((u, s) -> (Inf, 0.0), :nonfinite_field),
            )
            sol = trace(field, [0.0, 0.0]; direction = "forward")
            @test sol.status === status
            @test length(sol.points) == 1
            staged = (u, s) -> s == 0 ? (1.0, 0.0) : field(u, s)
            sol = trace(staged, [0.0, 0.0]; direction = "forward")
            @test sol.status === status
            @test length(sol.points) == 1
        end
        tiny = trace((u, s) -> (1.0e-100, 0.0), [0.0, 0.0];
            direction = "forward", maxstep = 2)
        @test tiny.status === :maxstep
        @test tiny.points[end] ≈ [0.01, 0.0]
        for invalid_call in 2:4
            calls = Ref(0)
            staged = (u, s) -> (calls[] += 1; calls[] == invalid_call ? (NaN, 0.0) : (1.0, 0.0))
            sol = trace(staged, [0.0, 0.0]; direction = "forward")
            @test sol.status === :nonfinite_field
            @test length(sol.points) == 1
            @test calls[] == invalid_call
        end
        overflow = trace((u, s) -> (1.0,), [floatmax(Float64)];
            direction = "forward", ds = floatmax(Float64), maxstep = 2)
        @test overflow.status === :nonfinite_state
        underflow = trace((u, s) -> (1.0,), [0.0];
            direction = "forward", ds = nextfloat(0.0), maxstep = 2)
        @test underflow.status === :stepsize_underflow
        @test_throws ErrorException trace((u, s) -> error("field error"), [0.0];
            direction = "forward")
        @test_throws ErrorException trace((u, s) -> (1.0,), [0.0];
            direction = "forward", isoutside = (u, s) -> error("predicate error"))
        @test_throws DimensionMismatch trace((u, s) -> (1.0,), [0.0, 0.0])
        @test_throws ArgumentError trace((u, s) -> (1.0im,), [0.0])
    end

    @testset "Input validation" begin
        for ds in (0.0, -0.1, NaN, Inf)
            @test_throws ArgumentError trace(straight, seed; ds)
        end
        for max_length in (-1.0, NaN, -Inf)
            @test_throws ArgumentError trace(straight, seed; max_length)
        end
        @test_throws ArgumentError trace(straight, seed; maxstep = 0)
        @test_throws ArgumentError trace(straight, seed; direction = "sideways")
        @test_throws ArgumentError trace(straight, seed; alg = Euler())
        @test_throws ArgumentError trace(straight, Float64[])
        @test_throws ArgumentError trace(straight, [NaN, 0.0, 0.0])
        @test_throws ArgumentError trace((u, s) -> (1.0f0,), Float32[0];
            ds = 0.1f0, max_length = nextfloat(0.0))
    end
end
