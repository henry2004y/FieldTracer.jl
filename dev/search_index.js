var documenterSearchIndex = {"docs":
[{"location":"internal/#API","page":"Internal","title":"API","text":"","category":"section"},{"location":"internal/#Public-APIs","page":"Internal","title":"Public APIs","text":"","category":"section"},{"location":"internal/","page":"Internal","title":"Internal","text":"Modules = [FieldTracer]\nPrivate = false\nOrder = [:constant, :type, :function]","category":"page"},{"location":"internal/#FieldTracer.select_seeds-Tuple{Any, Any, Any}","page":"Internal","title":"FieldTracer.select_seeds","text":" select_seeds(x, y, z; nsegment=(5, 5, 5))\n\nGenerate uniform seeding points in the grid range x, y and z in nsegment.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.select_seeds-Tuple{Any, Any}","page":"Internal","title":"FieldTracer.select_seeds","text":" select_seeds(x, y; nsegment=(5, 5))\n\nGenerate uniform seeding points in the grid range x and y in nsegment. If nsegment specified, use the keyword input, otherwise it will be overloaded by the 3D version seed generation function!\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace-Tuple","page":"Internal","title":"FieldTracer.trace","text":" trace(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;\n   alg=RK4(), kwargs...)\n\ntrace(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;\n\t alg=RK4(), maxstep=20000, ds=0.01, gridtype=\"ndgrid\", direction=\"both\")\n\nStream tracing on structured mesh with field in 3D array and grid in range.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace-Union{Tuple{TX}, Tuple{TV}, Tuple{Meshes.SimpleMesh, Vector{TV}, Vector{TV}, Vector{TX}, Vector{TX}}} where {TV<:AbstractFloat, TX<:AbstractFloat}","page":"Internal","title":"FieldTracer.trace","text":" trace(mesh::SimpleMesh, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)\n\n2D stream tracing on unstructured quadrilateral and triangular mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#Private-APIs","page":"Internal","title":"Private APIs","text":"","category":"section"},{"location":"internal/","page":"Internal","title":"Internal","text":"Modules = [FieldTracer]\nPublic = false","category":"page"},{"location":"internal/#FieldTracer.DoBreak-NTuple{4, Any}","page":"Internal","title":"FieldTracer.DoBreak","text":"DoBreak(iloc, jloc, iSize, jSize)\n\nCheck to see if we should break out of an integration.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.bilin_reg-NTuple{6, Any}","page":"Internal","title":"FieldTracer.bilin_reg","text":"bilin_reg(x, y, Q00, Q01, Q10, Q11)\n\nBilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1) Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.euler-NTuple{8, Any}","page":"Internal","title":"FieldTracer.euler","text":"euler(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)\n\nFast 2D tracing using Euler's method. It takes at most maxstep with step size ds tracing the vector field given by ux,uy starting from (startx,starty) in the Cartesian grid specified by ranges xGrid and yGrid. Step size is in normalized coordinates within the range [0, 1]. Return footprints' coordinates in (x, y).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.euler-Union{Tuple{T}, Tuple{Int64, T, T, T, T, Vararg{Any, 6}}} where T<:AbstractFloat","page":"Internal","title":"FieldTracer.euler","text":"euler(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)\n\nFast 3D tracing using Euler's method. It takes at most maxstep with step size ds tracing the vector field given by ux,uy,uz starting from  (startx,starty,startz) in the Cartesian grid specified by ranges xGrid, yGrid and zGrid. Return footprints' coordinates in (x,y,z).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.getCellID-Tuple{Meshes.SimpleMesh, Meshes.Point}","page":"Internal","title":"FieldTracer.getCellID","text":"Return cell ID on the unstructured mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.getelement-Tuple{Meshes.SimpleMesh, Int64}","page":"Internal","title":"FieldTracer.getelement","text":"Return the cellIDth element of the mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp-Tuple{Any, Any, Any, Any, Array}","page":"Internal","title":"FieldTracer.grid_interp","text":"grid_interp(x, y, field, ix, iy)\n\nInterpolate a value at (x,y) in a field. ix and iy are indexes for x,y locations (0-based).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp-Union{Tuple{U}, Tuple{T}, Tuple{T, T, T, Int64, Int64, Int64, AbstractArray{U, 3}}} where {T<:Real, U<:Number}","page":"Internal","title":"FieldTracer.grid_interp","text":"grid_interp(x, y, z, ix, iy, iz, field)\n\nInterpolate a value at (x,y,z) in a field. ix,iy and iz are indexes for x, y and z locations (0-based).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-NTuple{4, Any}","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-Union{Tuple{V}, Tuple{U}, Tuple{U, U, U, V, V, V}} where {U, V<:Real}","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field in normalized coordinates.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.rk4-NTuple{8, Any}","page":"Internal","title":"FieldTracer.rk4","text":"rk4(maxstep, ds, startx, starty, xGrid, yGrid, ux, uy)\n\nFast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and constant step size ds. See also euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.rk4-Union{Tuple{T}, Tuple{Int64, T, T, T, T, Vararg{Any, 6}}} where T<:AbstractFloat","page":"Internal","title":"FieldTracer.rk4","text":"rk4(maxstep, ds, startx, starty, startz, xGrid, yGrid, zGrid, ux, uy, uz)\n\nFast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and constant step size ds. See also euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_euler-NTuple{6, Any}","page":"Internal","title":"FieldTracer.trace2d_euler","text":" trace2d_euler(fieldx, fieldy, startx, starty, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridtype=\"ndgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method, which is faster but less accurate than RK4. Only valid for regular grid with coordinates' range gridx and gridy. Step size is in normalized coordinates within the range [0,1]. The field can be in both meshgrid or ndgrid (default) format. Supporting direction of {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_euler-Tuple{Any, Any, Any, Any, Meshes.CartesianGrid{M, C, N} where {M<:Meshes.𝔼, C<:CoordRefSystems.Cartesian, N}}","page":"Internal","title":"FieldTracer.trace2d_euler","text":"trace2d_euler(fieldx, fieldy, startx, starty, grid::CartesianGrid; kwargs...)\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_rk4-NTuple{6, Any}","page":"Internal","title":"FieldTracer.trace2d_rk4","text":" trace2d_rk4(fieldx, fieldy, startx, starty, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridtype=\"ndgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Runge Kutta 4. Slower than Euler, but more accurate. The higher accuracy allows for larger step sizes ds. Step size is in normalized coordinates within the range [0,1]. See also trace2d_euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_euler-Union{Tuple{T}, Tuple{Any, Any, Any, T, T, T, Any, Any, Any}} where T<:AbstractFloat","page":"Internal","title":"FieldTracer.trace3d_euler","text":" trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;\n   maxstep=20000, ds=0.01)\n\nGiven a 3D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. Only valid for regular grid with coordinates gridx, gridy, gridz. The field can be in both meshgrid or ndgrid (default) format. Supporting direction of {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_euler-Union{Tuple{T}, Tuple{F}, Tuple{F, F, F, T, T, T, Meshes.CartesianGrid{M, C, N} where {M<:Meshes.𝔼, C<:CoordRefSystems.Cartesian, N}}} where {F, T}","page":"Internal","title":"FieldTracer.trace3d_euler","text":"trace3d_euler(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;\n\t maxstep=20000, ds=0.01, gridtype=\"ndgrid\", direction=\"both\")\n\nSee also trace3d_rk4.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_rk4-Union{Tuple{T}, Tuple{Any, Any, Any, T, T, T, Any, Any, Any}} where T<:AbstractFloat","page":"Internal","title":"FieldTracer.trace3d_rk4","text":" trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, gridx, gridy, gridz;\n   maxstep=20000, ds=0.01)\n\nGiven a 3D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. Only valid for regular grid with coordinates gridx, gridy, gridz. The field can be in both meshgrid or ndgrid (default) format. See also trace3d_euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_rk4-Union{Tuple{T}, Tuple{F}, Tuple{F, F, F, T, T, T, Meshes.CartesianGrid{M, C, N} where {M<:Meshes.𝔼, C<:CoordRefSystems.Cartesian, N}}} where {F, T}","page":"Internal","title":"FieldTracer.trace3d_rk4","text":"trace3d_rk4(fieldx, fieldy, fieldz, startx, starty, startz, grid::CartesianGrid;\n\t maxstep=20000, ds=0.01, gridtype=\"ndgrid\", direction=\"both\")\n\nSee also trace3d_euler.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trilin_reg-Union{Tuple{T}, Tuple{T, T, T, Vararg{Any, 8}}} where T<:Real","page":"Internal","title":"FieldTracer.trilin_reg","text":"trilin_reg(x, y, z, Q)\n\nTrilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1) Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"log/#Log","page":"Log","title":"Log","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Streamline and trajectory are related topics in physical modeling, common seen in fluid and particle simulations.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"First make it work, then make it better and fast.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"One approach is called Pollock method.","category":"page"},{"location":"log/#Tracing-in-Unstructured-Grid","page":"Log","title":"Tracing in Unstructured Grid","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Given an unstructured grid with node points and connectivity, how should you do the streamline tracing?","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Brute force algorithm:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"find the grid cell you are currently in;\nmove along the vector direction until you hit the boundary of that cell;\nfind the neighbour who shares the same edge and the intersection point;\nuse the vector direction in the next cell and move along that direction;\nrepeat 2-4 until you reach any of the stopping criteria: hit the boundary, exceed MaxIteration, or exceed MaxLength.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Some questions during the process:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"How to find the neighbouring cell?\nHow to determine which boundary edge will you cross?\nHow to improve the search speed?\nHow to improve accuracy?","category":"page"},{"location":"log/#MATLAB","page":"Log","title":"MATLAB","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is an implementation of streamline tracing in Matlab called tristream. It requires nodal data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Inside the function, there is an intrinsic function called pointLocation, which returns the index of cell where the point locates.","category":"page"},{"location":"log/#YT","page":"Log","title":"YT","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is another implementation in yt library, which has many similarities to the one I borrowed from SpacePy.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Streamlining through a volume is useful for a variety of analysis tasks. By specifying a set of starting positions, the user is returned a set of 3D positions that can, in turn, be used to visualize the 3D path of the streamlines. Additionally, individual streamlines can be converted into YTStreamline objects, and queried for all the available fields along the streamline.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The implementation of streamlining in yt is described below.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Decompose the volume into a set of non-overlapping, fully domain tiling bricks, using the AMRKDTree homogenized volume.\nFor every streamline starting position:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"While the length of the streamline is less than the requested length:\nFind the brick that contains the current position.\nIf not already present, generate vertex-centered data for the vector fields defining the streamline.\nWhile inside the brick:\nintegrate the streamline path using a Runge-Kutta 4th order method and the vertex centered data.\nduring the intermediate steps of each RK4 step, if the position is updated to outside the current brick, interrupt the integration and locate a new brick at the intermediate position.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The set of streamline positions are stored in the Streamlines object.","category":"page"},{"location":"log/#VTK","page":"Log","title":"VTK","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"In the VTK library, there is a class called vtkPointLocator. It is a spatial search object to quickly locate points in 3D. vtkPointLocator works by dividing a specified region of space into a regular array of \"rectangular\" buckets, and then keeping a list of points that lie in each bucket. Typical operation involves giving a position in 3D and finding the closest point. It supports both nodal data and cell data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"vtkPointLocator has two distinct methods of interaction. In the first method, you supply it with a dataset, and it operates on the points in the dataset. In the second method, you supply it with an array of points, and the object operates on the array.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"note: Note\nMany other types of spatial locators have been developed such as octrees and kd-trees. These are often more efficient for the operations described here.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FieldTracer","category":"page"},{"location":"#FieldTracer","page":"Home","title":"FieldTracer","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Vector field tracing on common types of meshes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package supports 2nd and 4th order field line tracing on","category":"page"},{"location":"","page":"Home","title":"Home","text":"2D/3D regular Cartesian mesh\n2D unstructured quadrilateral mesh\n2D unstructured triangular mesh","category":"page"},{"location":"","page":"Home","title":"Home","text":"It can be used for","category":"page"},{"location":"","page":"Home","title":"Home","text":"generating streamlines and fieldlines;\nchecking field connectivity between spatial cells.","category":"page"},{"location":"example/#Examples","page":"Example","title":"Examples","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"There is one higher level function API, trace, for tracing on a 2D and 3D mesh. This function accepts different types of input mesh arguments.[1] The default scheme is 4th order Runge-Kutta method RK4(), but users can also switch to other options like 2nd order Euler method Euler().[2] User can also specify the maximum number of tracing steps via keyword maxstep, the step size in normalized coordinates [0,1] ds, order of grid gridtype, and tracing directions direction chosen between \"both\",\"forward\",\"backward\".","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"More examples can be found in the examples folder.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"[1]: Currently 3D tracing is only limited to structured mesh.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"[2]: The scheme switch currently only works for Cartesian grid.","category":"page"},{"location":"example/#Structured-2D-mesh","page":"Example","title":"Structured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"using FieldTracer\nx = range(0, 1, step=0.1)\ny = range(0, 1, step=0.1)\n# ndgrid\ngridx = [i for i in x, _ in y]\ngridy = [j for _ in x, j in y]\nu = copy(gridx)\nv = -copy(gridy)\nstartx = 0.1\nstarty = 0.9\ntrace(u, v, startx, starty, x, y)","category":"page"},{"location":"example/#Unstructured-2D-mesh","page":"Example","title":"Unstructured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"See the example in test_unstructured.jl.","category":"page"},{"location":"example/#Structured-3D-mesh","page":"Example","title":"Structured 3D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Tracing on a structured 3D mesh is a natural extension from 2D.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"using FieldTracer, Meshes\nx = range(0, 10, length=15)\ny = range(0, 10, length=20)\nz = range(0, 10, length=25)\nbx = fill(1.0, length(x), length(y), length(z))\nby = fill(1.0, length(x), length(y), length(z))\nbz = fill(1.0, length(x), length(y), length(z))\nxs, ys, zs = 1.0, 1.0, 1.0\nΔx = x[2] - x[1]\nΔy = y[2] - y[1]\nΔz = z[2] - z[1]\ngrid = CartesianGrid((length(x)-1, length(y)-1, length(z)-1),\n   (0., 0., 0.),\n   (Δx, Δy, Δz))\n\n# default direction is both\nx1, y1, z1 = trace(bx, by, bz, xs, ys, zs, grid; alg=Euler(), ds=0.2, maxstep=200)","category":"page"},{"location":"example/#Seeding","page":"Example","title":"Seeding","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"We provide a function select_seeds for generating uniform seeds for the starting points in 2D/3D. This ensures consistent sampling of fieldlines across the same points to reduce visual shift effect across multiple frames.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Furthermore, we can select seeds interactively on the figure and plot on-the-fly. See the example demo_interactive_select.jl.","category":"page"},{"location":"example/#Parallelization","page":"Example","title":"Parallelization","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Parallelized tracing can happen at fieldline level, whereas it can be less efficient to do it per line. We recommend providing a bunch of points to the trace function and then collecting results:","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"linex, liney = zeros(20,10), zeros(20,10)\nThreads.@threads for i = 1:10\n   lx, ly = trace(u, v, startx, starty, x, y, maxstep=20)\n   linex[1:length(lx),i] = lx\n   liney[1:length(ly),i] = ly\nend","category":"page"},{"location":"example/#Arrow","page":"Example","title":"Arrow","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"When plotting, it is usually convenient to display an arrow along the line for showing the direction. Currently we provide a function add_arrow which acts on a Matplotlib Line2D object and adds an arrow for it.[3]","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"[3]: 3D arrow is not supported due to the limitation of Matplotlib. It is possible, but tedious.","category":"page"},{"location":"example/#Gallery","page":"Example","title":"Gallery","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in an analytic asymptotic field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a analytic 3D dipole field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a numerical 2D magnetic field","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Tracing in a 2D equatorial plane Earth magnetosphere simulation","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Streamline tracing in a 2D triangular mesh around an airfoil","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"An example is shown for the 2D streamline tracing in the unstructured triangular mesh for the famous airfoil problem. The blue lines are the analytic stream functions derived from incompressible Euler equations which are calculated numerically. Three colored lines are displayed with dots representing the footprints inside each cell.","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Fieldline tracing near the magnetic null points, compared to the streamplot function in Matplotlib","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Fieldline tracing near Ganymede's upstream magnetopause","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"}]
}
