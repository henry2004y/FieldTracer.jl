using FieldTracer, Meshes
using Test

@testset "FieldTracer.jl" begin
   @testset "Seeding" begin
      x = 0.0:0.1:1.0
      y = 1.0:0.1:2.0
      seeds = select_seeds(x, y; nSeed=4)
      @test seeds[:,4] ≈ [0.177329, 1.00791] atol=1e-5
      z = 2.0:0.1:3.0
      seeds = select_seeds(x, y, z; nSeed=2)
      @test seeds[:,2] ≈ [0.910357, 1.34652, 2.52388] atol=1e-5
   end

   @testset "2D structured mesh" begin
      # bilinear interpolation function in a normalized rectangle
      x, y = 0.1, 0.2
      Q00, Q10, Q01, Q11 = 3.0, 40.0, 5.0, 60.0
      out = FieldTracer.bilin_reg(x, y, Q00, Q10, Q01, Q11)
      @test out ≈ 7.460 atol=1e-5

      # Euler2 and RK4 functions
      # https://se.mathworks.com/help/matlab/ref/streamline.html
      ds, maxstep = 0.1, 150
      x = range(0, 1, step=0.1)
      y = range(0, 1, step=0.1)
      # ndgrid
      xgrid = [i for i in x, _ in y]
      ygrid = [j for _ in x, j in y]
      u = copy(xgrid)
      v = -copy(ygrid)
      xstart = 0.1
      ystart = 0.9

      xt, yt = FieldTracer.euler(maxstep, ds, xstart, ystart, x, y, u, v)

      @test length(xt) == 141

      xt, yt = FieldTracer.rk4(maxstep, ds, xstart, ystart, x, y, u, v)

      @test length(xt) == 140

      grid = CartesianGrid((length(x)-1,length(y)-1),(0.,0.),(0.1,0.1))

      # default direction is both
      xt, yt = trace(u, v, xstart, ystart, grid; alg=Euler(), maxstep, ds)

      @test length(xt) == 148

      include("test_structured.jl")
      # asymptotic field
      @test test_trace_asymptote()     # single precision
      @test test_trace_asymptote(true) # double precision

      # dipole field
      @test test_trace_dipole()        # dipole tracing in 2D
   end

   @testset "3D structured mesh" begin
      # trilinear interpolation function in a normalized box
      x, y, z = 0.1, 0.2, 0.3
      Q = [3.0, 40.0, 5.0, 60.0, 3.0, 40.0, 5.0, 60.0]
      out = FieldTracer.trilin_reg(x, y, z, Q)
      @test out ≈ 7.460 atol=1e-5

      x = range(0, 10, length=15)
      y = range(0, 10, length=20)
      z = range(0, 10, length=25)
      bx = fill(1.0, length(x), length(y), length(z))
      by = fill(1.0, length(x), length(y), length(z))
      bz = fill(1.0, length(x), length(y), length(z))
      xs, ys, zs = 1.0, 1.0, 1.0
   
      # Euler 2nd order
      x1, y1, z1 = trace(bx, bz, bz, xs, ys, zs, x, y, z;
         alg=Euler(), ds=0.2, maxstep=200)

      @test length(x1) == 170 && x1[end] ≈ y1[end] ≈ z1[end]

      # RK4 by default
      x1, y1, z1 = trace(bx, bz, bz, xs, ys, zs, x, y, z;
         ds=0.2, maxstep=200, direction="forward")

      @test length(x1) == 152 && x1[end] ≈ y1[end] ≈ z1[end]

      Δx = x[2] - x[1]
      Δy = y[2] - y[1]
      Δz = z[2] - z[1]

      grid = CartesianGrid((length(x)-1, length(y)-1, length(z)-1),
         (0., 0., 0.),
         (Δx, Δy, Δz))

      # default direction is both
      x1, y1, z1 = trace(bx, bz, bz, xs, ys, zs, grid;
         alg=Euler(), ds=0.2, maxstep=200)

      @test length(x1) == 170 && x1[end] ≈ y1[end] ≈ z1[end]
   end

   @testset "2D unstructured mesh" begin
      include("test_unstructured.jl")
      @test test_trace_unstructured2D()
   end

end
