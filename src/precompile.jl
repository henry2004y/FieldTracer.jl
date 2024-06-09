# Precompiling workloads

@setup_workload begin
   @compile_workload begin
      x = range(0, 1, step=0.1)
      y = range(0, 1, step=0.1)
      # ndgrid
      u, v = let
         xgrid = [i for i in x, _ in y]
         ygrid = [j for _ in x, j in y]
         copy(xgrid), -copy(ygrid)
      end
      xstart = 0.1
      ystart = 0.9

      #xt, yt = FieldTracer.euler(150, 0.1, 0.1, 0.9, x, y, u, v)
      xt, yt = trace(u, v, xstart, ystart, x, y;
         maxstep=100, ds=0.1, direction="both", alg=Euler())
      xt, yt = trace(u, v, xstart, ystart, x, y;
         maxstep=100, ds=0.1, direction="both", alg=RK4())

      x = range(0, 10, length=10)
      y = range(0, 10, length=10)
      z = range(0, 10, length=10)
      bx = fill(1.0, length(x), length(y), length(z))
      by = fill(1.0, length(x), length(y), length(z))
      bz = fill(1.0, length(x), length(y), length(z))
      xs, ys, zs = 1.0, 1.0, 1.0
   
      # Euler 2nd order
      x1, y1, z1 = trace(bx, by, bz, xs, ys, zs, x, y, z;
         maxstep=10, ds=0.2, direction="both", alg=Euler())
      # RK4
      x1, y1, z1 = trace(bx, by, bz, xs, ys, zs, x, y, z;
         maxstep=10, ds=0.2, direction="both", alg=RK4())
   end
end