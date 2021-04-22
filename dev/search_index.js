var documenterSearchIndex = {"docs":
[{"location":"internal/","page":"Internal","title":"Internal","text":"Modules = [FieldTracer]","category":"page"},{"location":"internal/#FieldTracer.DoBreak-NTuple{4, Any}","page":"Internal","title":"FieldTracer.DoBreak","text":"DoBreak(iloc, jloc, iSize, jSize)\n\nCheck to see if we should break out of an integration.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.Euler-NTuple{11, Any}","page":"Internal","title":"FieldTracer.Euler","text":"Euler(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)\n\nFast 3D tracing using Euler's method. It takes at most maxstep with step size ds tracing the vector field given by ux,uy,uz starting from  (startx,starty,startz) in the Cartesian grid specified by ranges xGrid, yGrid and zGrid. Return footprints' coordinates in (x,y,z).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.Euler-NTuple{8, Any}","page":"Internal","title":"FieldTracer.Euler","text":"Euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)\n\nFast 2D tracing using Euler's method. It takes at most maxstep with step size ds tracing the vector field given by ux,uy starting from (startx,starty) in the Cartesian grid specified by ranges xGrid and yGrid. Return footprints' coordinates in (x,y).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.RK4-NTuple{11, Any}","page":"Internal","title":"FieldTracer.RK4","text":"RK4(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)\n\nFast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and constant step size ds. See also Euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.RK4-NTuple{8, Any}","page":"Internal","title":"FieldTracer.RK4","text":"RK4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)\n\nFast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and constant step size ds. See also Euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.add_arrow","page":"Internal","title":"FieldTracer.add_arrow","text":"Add an arrow to the object line from Matplotlib.\n\n\n\n\n\n","category":"function"},{"location":"internal/#FieldTracer.bilin_reg-NTuple{6, Any}","page":"Internal","title":"FieldTracer.bilin_reg","text":"bilin_reg(x, y, Q00, Q01, Q10, Q11)\n\nBilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1) Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.getCellID-Tuple{Meshes.SimpleMesh, Meshes.Point2}","page":"Internal","title":"FieldTracer.getCellID","text":"getCellID(mesh::SimpleMesh, x, y)\n\nReturn cell ID on the unstructured mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp-NTuple{7, Any}","page":"Internal","title":"FieldTracer.grid_interp","text":"grid_interp(x, y, z, field, ix, iy, iz, xsize, ysize)\n\nInterpolate a value at (x,y,z) in a field. ix,iy and iz are indexes for x,y and z locations (0-based). xsize and ysize are the sizes of field in X and Y.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp-Tuple{Any, Any, Array, Any, Any}","page":"Internal","title":"FieldTracer.grid_interp","text":"grid_interp(x, y, field, ix, iy)\n\nInterpolate a value at (x,y) in a field. ix and iy are indexes for x,y locations (0-based).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-NTuple{4, Any}","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-NTuple{6, Any}","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field in normalized coordinates.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.select_seeds-Tuple{Any, Any}","page":"Internal","title":"FieldTracer.select_seeds","text":" select_seeds(x, y; nSeed=100)\n\nGenerate nSeed seeding points randomly in the grid range x and y. If you specify nSeed, use the keyword input, otherwise it will be overloaded by the 3D version seed generation function!\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace-NTuple{6, Any}","page":"Internal","title":"FieldTracer.trace","text":" trace(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)\n\n2D stream tracing on structured mesh with field in 2D array and grid in range.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace-NTuple{9, Any}","page":"Internal","title":"FieldTracer.trace","text":" trace(fieldx, fieldy, startx, starty, gridx, gridy; kwargs...)\n\n3D stream tracing on structured mesh with field in 3D array and grid in range.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace-Tuple{Meshes.SimpleMesh, Any, Any, Any, Any}","page":"Internal","title":"FieldTracer.trace","text":" trace(mesh::SimpleMesh, vx, vy, xstart, ystart;\n   maxIter=1000, maxLen=1000.)\n\n2D stream tracing on unstructured quadrilateral and triangular mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_euler-NTuple{6, Any}","page":"Internal","title":"FieldTracer.trace2d_euler","text":" trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridType=\"ndgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method, which is faster but less accurate than RK4. Only valid for regular grid with coordinates' range gridx and gridy. The field can be in both meshgrid or ndgrid (default) format. Supporting direction of {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_euler-Tuple{Any, Any, Any, Any, Meshes.CartesianGrid}","page":"Internal","title":"FieldTracer.trace2d_euler","text":"trace2d_euler(fieldx, fieldy, startx, starty, grid::CartesianGrid;\n   kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_rk4-NTuple{6, Any}","page":"Internal","title":"FieldTracer.trace2d_rk4","text":" trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridType=\"ndgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Runge Kutta 4. Slower than Euler, but more accurate. The higher accuracy allows for larger step sizes ds. See also trace2d_euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_euler-NTuple{9, Any}","page":"Internal","title":"FieldTracer.trace3d_euler","text":" trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy,\n   gridz; maxstep=20000, ds=0.01)\n\nGiven a 3D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. Only valid for regular grid with coordinates gridx, gridy, gridz. The field can be in both meshgrid or ndgrid (default) format. Supporting direction of {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_euler-Tuple{Any, Any, Any, Any, Any, Any, Meshes.CartesianGrid}","page":"Internal","title":"FieldTracer.trace3d_euler","text":"trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz,\n   grid::CartesianGrid;\n\t maxstep=20000, ds=0.01, gridType=\"ndgrid\", direction=\"both\")\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_rk4-NTuple{9, Any}","page":"Internal","title":"FieldTracer.trace3d_rk4","text":" trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy,\n   gridz; maxstep=20000, ds=0.01)\n\nGiven a 3D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. Only valid for regular grid with coordinates gridx, gridy, gridz. The field can be in both meshgrid or ndgrid (default) format.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trilin_reg-NTuple{4, Any}","page":"Internal","title":"FieldTracer.trilin_reg","text":"trilin_reg(x, y, z, Q)\n\nTrilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1) Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"log/#Log","page":"Log","title":"Log","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Streamline and trajectory are related topics in physical modeling, common seen in fluid and particle simulations.","category":"page"},{"location":"log/#Streamline-Tracing","page":"Log","title":"Streamline Tracing","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"First make it work, then make it better and fast.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"I found an approach called Pollock method.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"I need an adaptive step control integration scheme like rk45. From DifferentialEquations.jl, there are hundreds of schemes we can choose from, which are much better than implementing our own fragile ODE solver!","category":"page"},{"location":"log/#Tracing-in-Unstructured-Grid","page":"Log","title":"Tracing in Unstructured Grid","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Given an unstructured grid with node points and connectivity, how should you do the streamline tracing?","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Brute force algorithm:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"find the grid cell you are currently in;\nmove along the vector direction until you hit the boundary of that cell;\nfind the neighbour who shares the same edge and the intersection point;\nuse the vector direction in the next cell and move along that direction;\nrepeat 2-4 until you reach any of the stopping criteria: hit the boundary, exceed MaxIteration, or exceed MaxLength.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Some questions during the process:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"How to find the neighbouring cell?\nHow to determine which boundary edge will you cross?\nHow to improve the search speed?\nHow to improve accuracy?","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"A package UnstructuredGrids.jl already exists. I take advantage of this package and quickly build a 2D stream tracer on unstructured 2D grid based on the brute force algorithm. Unfortunately it is no longer maintained. I substituted it with Meshes.jl, which provides a cleaner interface and seems to be carefully designed.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Actually, the brute force algorithm may not be as bad as you think in terms of accuracy. Finite volume method uses one value per cell to represent the solution space, therefore it is just cheating to use higher order method for stream tracing.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"An example is shown for the 2D streamline tracing in the unstructured triangular mesh for the famous airfoil problem. The blue lines are the analytic stream functions derived from incompressible Euler equations which are calculated numerically. Three colored lines are displayed with dots representing the footprint inside each cell.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"An extension to 3D is possible and I may work on it in the future.","category":"page"},{"location":"log/#MATLAB","page":"Log","title":"MATLAB","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is an implementation of streamline tracing in Matlab called tristream. It requires nodal data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Inside the function, there is an intrinsic function called pointLocation, which returns the index of cell where the point locates.","category":"page"},{"location":"log/#yt","page":"Log","title":"yt","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is another implementation in yt library, which has many similarities to the one I borrowed from SpacePy.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Streamlining through a volume is useful for a variety of analysis tasks. By specifying a set of starting positions, the user is returned a set of 3D positions that can, in turn, be used to visualize the 3D path of the streamlines. Additionally, individual streamlines can be converted into YTStreamline objects, and queried for all the available fields along the streamline.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The implementation of streamlining in yt is described below.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Decompose the volume into a set of non-overlapping, fully domain tiling bricks, using the AMRKDTree homogenized volume.\nFor every streamline starting position:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"While the length of the streamline is less than the requested length:\nFind the brick that contains the current position\nIf not already present, generate vertex-centered data for the vector fields defining the streamline.\nWhile inside the brick\nIntegrate the streamline path using a Runge-Kutta 4th order method and the vertex centered data.\nDuring the intermediate steps of each RK4 step, if the position is updated to outside the current brick, interrupt the integration and locate a new brick at the intermediate position.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The set of streamline positions are stored in the Streamlines object.","category":"page"},{"location":"log/#VTK","page":"Log","title":"VTK","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"In the VTK library, there is a class called vtkPointLocator. It is a spatial search object to quickly locate points in 3D. vtkPointLocator works by dividing a specified region of space into a regular array of \"rectangular\" buckets, and then keeping a list of points that lie in each bucket. Typical operation involves giving a position in 3D and finding the closest point. It supports both nodal data and cell data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"vtkPointLocator has two distinct methods of interaction. In the first method, you supply it with a dataset, and it operates on the points in the dataset. In the second method, you supply it with an array of points, and the object operates on the array.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"note: Note\nMany other types of spatial locators have been developed such as octrees and kd-trees. These are often more efficient for the operations described here.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FieldTracer","category":"page"},{"location":"#FieldTracer","page":"Home","title":"FieldTracer","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Vector field tracing on common types of meshes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package supports 2nd order and 4th order field line tracing on","category":"page"},{"location":"","page":"Home","title":"Home","text":"2D/3D regular Cartesian mesh\n2D unstructured quadrilateral mesh\n2D unstructured triangular mesh","category":"page"},{"location":"example/#Examples","page":"Example","title":"Examples","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"There are two higher level function APIs, trace2d for tracing on a 2D mesh and trace3d for tracing on a 3D mesh. These two functions accept different types of input mesh arguments. Currently 3D tracing is only limited to structured mesh.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"More examples can be found in the examples folder.","category":"page"},{"location":"example/#Structured-2D-mesh","page":"Example","title":"Structured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We provide 2nd order Euler method trace2d_euler and 4th order Runge-Kutta method (default) trace2d_rk4 for tracing on a structured mesh.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"using FieldTracer\nx = range(0, 1, step=0.1)\ny = range(0, 1, step=0.1)\n# ndgrid\ngridx = [i for i in x, _ in y]\ngridy = [j for _ in x, j in y]\nu = copy(gridx)\nv = -copy(gridy)\nstartx = 0.1\nstarty = 0.9\ntrace2d(u, v, startx, starty, gridx, gridy)","category":"page"},{"location":"example/#Unstructured-2D-mesh","page":"Example","title":"Unstructured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"See the example in test_unstructured.jl.","category":"page"},{"location":"example/#Structured-3D-mesh","page":"Example","title":"Structured 3D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Tracing on a structured 3D mesh is a natural extension from 2D.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"using FieldTracer, Meshes\nx = range(0, 10, length=15)\ny = range(0, 10, length=20)\nz = range(0, 10, length=25)\nbx = fill(1.0, length(x), length(y), length(z))\nby = fill(1.0, length(x), length(y), length(z))\nbz = fill(1.0, length(x), length(y), length(z))\nxs, ys, zs = 1.0, 1.0, 1.0\nΔx = x[2] - x[1]\nΔy = y[2] - y[1]\nΔz = z[2] - z[1]\ngrid = CartesianGrid((length(x)-1, length(y)-1, length(z)-1),\n   (0., 0., 0.),\n   (Δx, Δy, Δz))\n\n# default direction is both\nx1, y1, z1 = trace3d_euler(bx, bz, bz, xs, ys, zs, grid, ds=0.2, maxstep=200)","category":"page"},{"location":"example/#Seeding","page":"Example","title":"Seeding","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We provide a function select_seeds for generating pseudo random seeds for the starting points in 2D/3D. This ensures consistent sampling of fieldlines across the same points to reduce visual shift effect across multiple frames.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Furthermore, we can select seeds interactively on the figure and plot on-the-fly. See the example demointeractiveselect.jl.","category":"page"},{"location":"example/#Arrow","page":"Example","title":"Arrow","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"When plotting, it is usually convenient to display an arrow along the line for showing the direction. Currently we provide a function add_arrow which acts on a Matplotlib Line2D object and adds an arrow for it.","category":"page"},{"location":"example/#Gallery","page":"Example","title":"Gallery","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in an analytic asymptotic field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a analytic 3D dipole field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a numerical 2D magnetic field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a 2D equatorial plane Earth magnetosphere simulation","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Streamline tracing in a 2D triangular mesh around an airfoil","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Fieldline tracing near the magnetic null points, compared to the streamplot function in Matplotlib","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"}]
}
