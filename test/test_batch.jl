using FieldTracer
using Test
using Test

@testset "Batched Tracing" begin
    @testset "2D Batch vs Serial (Forward)" begin
        x = 0.0:0.1:10.0
        y = 0.0:0.1:10.0
        ux = rand(length(x), length(y))
        uy = rand(length(x), length(y))

        N = 10
        startx = rand(N) .* 10.0
        starty = rand(N) .* 10.0
        maxstep = 50
        ds = 0.05

        # Euler
        xt_batch, yt_batch = FieldTracer.trace2d_euler(ux, uy, startx, starty, x, y; maxstep, ds, direction = "forward")

        @test size(xt_batch) == (N, maxstep)

        # Verify against serial
        for i in 1:N
            xt_s, yt_s = FieldTracer.trace2d_euler(ux, uy, startx[i], starty[i], x, y; maxstep, ds, direction = "forward")
            len_s = length(xt_s)

            # Vectorized comparison for the active part
            @test xt_batch[i, 1:len_s] ≈ xt_s atol = 0.2
            @test yt_batch[i, 1:len_s] ≈ yt_s atol = 0.2

            # Check constant padding
            if len_s < maxstep
                @test all(≈(xt_s[end]; atol = 0.2), @view xt_batch[i, (len_s + 1):end])
                @test all(≈(yt_s[end]; atol = 0.2), @view yt_batch[i, (len_s + 1):end])
            end
        end

        # RK4
        xt_batch, yt_batch = FieldTracer.trace2d_rk4(ux, uy, startx, starty, x, y; maxstep, ds, direction = "forward")

        for i in 1:N
            xt_s, yt_s = FieldTracer.trace2d_rk4(ux, uy, startx[i], starty[i], x, y; maxstep, ds, direction = "forward")
            len_s = length(xt_s)

            # Vectorized comparison
            @test xt_batch[i, 1:len_s] ≈ xt_s atol = 0.2
            @test yt_batch[i, 1:len_s] ≈ yt_s atol = 0.2

            if len_s < maxstep
                @test all(≈(xt_s[end]; atol = 0.2), @view xt_batch[i, (len_s + 1):end])
                @test all(≈(yt_s[end]; atol = 0.2), @view yt_batch[i, (len_s + 1):end])
            end
        end
    end

    @testset "3D Batch vs Serial (Forward)" begin
        x = 0.0:0.1:5.0
        y = 0.0:0.1:5.0
        z = 0.0:0.1:5.0
        ux = rand(length(x), length(y), length(z))
        uy = rand(length(x), length(y), length(z))
        uz = rand(length(x), length(y), length(z))

        N = 5
        startx = rand(N) .* 5.0
        starty = rand(N) .* 5.0
        startz = rand(N) .* 5.0
        maxstep = 20
        ds = 0.05

        # Euler
        xt_batch, yt_batch, zt_batch = FieldTracer.trace3d_euler(ux, uy, uz, startx, starty, startz, x, y, z; maxstep, ds, direction = "forward")

        @test size(xt_batch) == (N, maxstep)

        for i in 1:N
            xt_s, yt_s, zt_s = FieldTracer.trace3d_euler(ux, uy, uz, startx[i], starty[i], startz[i], x, y, z; maxstep, ds, direction = "forward")
            len_s = length(xt_s)

            # Vectorized comparison
            @test xt_batch[i, 1:len_s] ≈ xt_s atol = 0.2
            @test yt_batch[i, 1:len_s] ≈ yt_s atol = 0.2
            @test zt_batch[i, 1:len_s] ≈ zt_s atol = 0.2

            if len_s < maxstep
                @test all(≈(xt_s[end]; atol = 0.2), @view xt_batch[i, (len_s + 1):end])
                @test all(≈(yt_s[end]; atol = 0.2), @view yt_batch[i, (len_s + 1):end])
                @test all(≈(zt_s[end]; atol = 0.2), @view zt_batch[i, (len_s + 1):end])
            end
        end

        # RK4
        xt_batch, yt_batch, zt_batch = FieldTracer.trace3d_rk4(ux, uy, uz, startx, starty, startz, x, y, z; maxstep, ds, direction = "forward")

        for i in 1:N
            xt_s, yt_s, zt_s = FieldTracer.trace3d_rk4(ux, uy, uz, startx[i], starty[i], startz[i], x, y, z; maxstep, ds, direction = "forward")
            len_s = length(xt_s)

            # Vectorized comparison
            @test xt_batch[i, 1:len_s] ≈ xt_s atol = 0.2
            @test yt_batch[i, 1:len_s] ≈ yt_s atol = 0.2
            @test zt_batch[i, 1:len_s] ≈ zt_s atol = 0.2

            if len_s < maxstep
                @test all(≈(xt_s[end]; atol = 0.2), @view xt_batch[i, (len_s + 1):end])
                @test all(≈(yt_s[end]; atol = 0.2), @view yt_batch[i, (len_s + 1):end])
                @test all(≈(zt_s[end]; atol = 0.2), @view zt_batch[i, (len_s + 1):end])
            end
        end
    end
end
