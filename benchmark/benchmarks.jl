# Benchmark Test for FieldTracer

using BenchmarkTools
using FieldTracer

const SUITE = BenchmarkGroup()

SUITE["trace"] = BenchmarkGroup()
SUITE["trace"]["2D structured"] = BenchmarkGroup()
SUITE["trace"]["3D structured"] = BenchmarkGroup()

# Euler2 and RK4 functions
ds, maxstep = 0.1, 100
x = range(0, 1, step=0.1)
y = range(0, 1, step=0.1)
# ndgrid
xgrid = [i for i in x, _ in y]
ygrid = [j for _ in x, j in y]
u = copy(xgrid)
v = -copy(ygrid)
xstart = 0.1
ystart = 0.9

SUITE["trace"]["2D structured"]["euler"] =
   @benchmarkable trace($u, $v, $xstart, $ystart, $x, $y;
      alg=Euler(), ds=$ds, maxstep=$maxstep)

SUITE["trace"]["2D structured"]["rk4"] =
   @benchmarkable trace($u, $v, $xstart, $ystart, $x, $y;
      alg=RK4(), ds=$ds, maxstep=$maxstep)

x = range(0, 10, length=15)
y = range(0, 10, length=20)
z = range(0, 10, length=25)
bx = fill(1.0, length(x), length(y), length(z))
by = fill(1.0, length(x), length(y), length(z))
bz = fill(1.0, length(x), length(y), length(z))
xs, ys, zs = 1.0, 1.0, 1.0

SUITE["trace"]["3D structured"]["euler"] =
   @benchmarkable trace($bx, $by, $bz, $xs, $ys, $zs, $x, $y, $z;
      alg=Euler(), ds=$ds, maxstep=$maxstep)
SUITE["trace"]["3D structured"]["rk4"] =
   @benchmarkable trace($bx, $by, $bz, $xs, $ys, $zs, $x, $y, $z; ds=$ds, maxstep=$maxstep)
