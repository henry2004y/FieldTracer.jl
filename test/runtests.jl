using FieldTracer
using Test

@testset "FieldTracer.jl" begin
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
      startx = 0.1
      starty = 0.9
      xt = Vector{Float64}(undef, maxstep) # output x
      yt = Vector{Float64}(undef, maxstep) # output y

      npoints = FieldTracer.Euler!(length(x), length(y), maxstep, ds,
         startx, starty, x, y, u, v, xt, yt)

      @test npoints == 141

      npoints = FieldTracer.RK4!(length(x), length(y), maxstep, ds,
         startx, starty, x, y, u, v, xt, yt)

      @test npoints == 140

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
      X = [i for i in x, _ in y, _ in z]
      Y = [j for _ in x, j in y, _ in z]
      Z = [k for _ in x, _ in y, k in z]
      bx, by, bz = similar(X), similar(X), similar(X)
      bx .= 1.0; by .= 1.0; bz .= 1.0
      xs, ys, zs = 1.0, 1.0, 1.0
   
      # Euler2
      x1, y1, z1 = trace3d_eul(bx, bz, bz, xs, ys, zs, x, y, z, ds=0.2, maxstep=200)

      @test length(x1) == 153 && x1[end] ≈ y1[end] ≈ z1[end]
   end

   @testset "2D unstructured mesh" begin
      include("test_unstructured.jl")
      @test test_trace_unstructured2D()
   end

end
