# Benchmark Test for FieldTracer

using BenchmarkTools
using FieldTracer

const SUITE = BenchmarkGroup()

SUITE["trace"] = BenchmarkGroup()
SUITE["trace"]["2D structured"] = BenchmarkGroup()
SUITE["trace"]["3D structured"] = BenchmarkGroup()

# Euler2 and RK4 functions
ds, maxstep = 0.1, 100
x = range(0, 1, step = 0.1)
y = range(0, 1, step = 0.1)
# ndgrid
xgrid = [i for i in x, _ in y]
ygrid = [j for _ in x, j in y]
u = copy(xgrid)
v = -copy(ygrid)
xstart = 0.1
ystart = 0.9

SUITE["trace"]["2D structured"]["euler"] =
    @benchmarkable trace(
    $u, $v, $xstart, $ystart, $x, $y;
    alg = Euler(), ds = $ds, maxstep = $maxstep
)

SUITE["trace"]["2D structured"]["rk4"] =
    @benchmarkable trace(
    $u, $v, $xstart, $ystart, $x, $y;
    alg = RK4(), ds = $ds, maxstep = $maxstep
)

x = range(0, 10, length = 15)
y = range(0, 10, length = 20)
z = range(0, 10, length = 25)
bx = fill(1.0, length(x), length(y), length(z))
by = fill(1.0, length(x), length(y), length(z))
bz = fill(1.0, length(x), length(y), length(z))
xs, ys, zs = 1.0, 1.0, 1.0

SUITE["trace"]["3D structured"]["euler"] =
    @benchmarkable trace(
    $bx, $by, $bz, $xs, $ys, $zs, $x, $y, $z;
    alg = Euler(), ds = $ds, maxstep = $maxstep
)
SUITE["trace"]["3D structured"]["rk4"] =
    @benchmarkable trace($bx, $by, $bz, $xs, $ys, $zs, $x, $y, $z; ds = $ds, maxstep = $maxstep)

# Batch Tracing Setup
nbatch = 1000

# 2D Batch
startx_batch_2d = rand(nbatch)
starty_batch_2d = rand(nbatch)

SUITE["trace"]["2D structured"]["batch_euler"] =
    @benchmarkable trace(
    $u, $v, $startx_batch_2d, $starty_batch_2d, $x, $y;
    alg = Euler(), ds = 0.01, maxstep = 100
)

SUITE["trace"]["2D structured"]["batch_rk4"] =
    @benchmarkable trace(
    $u, $v, $startx_batch_2d, $starty_batch_2d, $x, $y;
    alg = RK4(), ds = 0.01, maxstep = 100
)

# 3D Batch
startx_batch_3d = rand(nbatch) .* 10.0
starty_batch_3d = rand(nbatch) .* 10.0
startz_batch_3d = rand(nbatch) .* 10.0

SUITE["trace"]["3D structured"]["batch_euler"] =
    @benchmarkable trace(
    $bx, $by, $bz, $startx_batch_3d, $starty_batch_3d, $startz_batch_3d, $x, $y, $z;
    alg = Euler(), ds = 0.01, maxstep = 100
)

SUITE["trace"]["3D structured"]["batch_rk4"] =
    @benchmarkable trace(
    $bx, $by, $bz, $startx_batch_3d, $starty_batch_3d, $startz_batch_3d, $x, $y, $z;
    alg = RK4(), ds = 0.01, maxstep = 100
)

SUITE["trace"]["2D unstructured"] = BenchmarkGroup()

# Create a simple mesh for unstructured benchmark
using Meshes
grid = CartesianGrid(100, 100) # 10000 elements
mesh = convert(SimpleMesh, grid)
vx = fill(0.5, nelements(mesh))
vy = fill(0.5, nelements(mesh))
start_u = [50.5]
start_v = [50.5]

SUITE["trace"]["2D unstructured"]["trace"] =
    @benchmarkable trace($mesh, $vx, $vy, $start_u, $start_v)
