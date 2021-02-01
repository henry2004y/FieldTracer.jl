var documenterSearchIndex = {"docs":
[{"location":"internal/","page":"Internal","title":"Internal","text":"Modules = [FieldTracer]","category":"page"},{"location":"internal/#FieldTracer.DoBreak-Union{Tuple{T}, NTuple{4,T}} where T<:Integer","page":"Internal","title":"FieldTracer.DoBreak","text":"DoBreak(iloc, jloc, iSize, jSize)\n\nCheck to see if we should break out of an integration.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.Euler!-NTuple{12,Any}","page":"Internal","title":"FieldTracer.Euler!","text":"Euler!(iSize,jSize, maxstep, ds, xstart,ystart, xGrid,yGrid, ux,uy, x,y)\n\nSimple 2D tracing using Euler's method. Super fast but not super accurate.\n\nArguments\n\niSize::Int,jSize::Int: grid size.\nmaxstep::Int: max steps.\nds::Float64: step size.\nxstart::Float64, ystart::Float64: starting location.\nxGrid::Vector{Float64},yGrid::Vector{Float64}: actual coord system.\nux::Array{Float64,2},uy::Array{Float64,2}: field to trace through.\nx::Vector{Float64},y::Vector{Float64}: x, y of result stream.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.Euler!-NTuple{17,Any}","page":"Internal","title":"FieldTracer.Euler!","text":"Euler!(iSize, jSize, kSize, maxstep, ds, xstart, ystart, zstart,\n  xGrid, yGrid, zGrid, ux, uy, uz, x, y, z)\n\nSimple 3D tracing using Euler's method.\n\nArguments\n\niSize::Int,jSize::Int,kSize::Int: grid size.\nmaxstep::Int: max steps.\nds::Float64: step size.\nxstart::Float64, ystart::Float64, zstart::Float64: starting location.\nxGrid::Array{Float64,2},yGrid::Array{Float64,2},zGrid::Array{Float64,2}: actual coord system.\nux::Array{Float64,2},uy::Array{Float64,2},uz::Array{Float64,2}: field to trace through.\nx::Vector{Float64},y::Vector{Float64},z::Vector{Float64}: x, y, z of result stream.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.RK4!-NTuple{12,Any}","page":"Internal","title":"FieldTracer.RK4!","text":"RK4!(iSize,jSize, maxstep, ds, xstart,ystart, xGrid,yGrid, ux,uy, x,y)\n\nFast and reasonably accurate 2D tracing with 4th order Runge-Kutta method and constant step size ds.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.RK4!-NTuple{17,Any}","page":"Internal","title":"FieldTracer.RK4!","text":"RK4!(iSize,jSize,kSize, maxstep, ds, xstart,ystart,zstart,\n  xGrid,yGrid,zGrid, ux,uy,uz, x,y,z)\n\nFast and reasonably accurate 3D tracing with 4th order Runge-Kutta method and constant step size ds.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.bilin_reg-NTuple{6,Any}","page":"Internal","title":"FieldTracer.bilin_reg","text":"bilin_reg(x, y, Q00, Q01, Q10, Q11)\n\nBilinear interpolation for x1,y1=(0,0) and x2,y2=(1,1) Q's are surrounding points such that Q00 = F[0,0], Q10 = F[1,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.getCellID-Tuple{Meshes.UnstructuredMesh,Meshes.Point{2,Float64}}","page":"Internal","title":"FieldTracer.getCellID","text":"getCellID(mesh::UnstructuredMesh, x, y)\n\nReturn cell ID in the unstructured grid.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp!-NTuple{7,Any}","page":"Internal","title":"FieldTracer.grid_interp!","text":"grid_interp!(x, y, z, field, ix, iy, iz, xsize, ysize)\n\nInterpolate a value at (x,y,z) in a field. ix,iy and iz are indexes for x,y and z locations (0-based). xsize and ysize are the sizes of field in X and Y.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.grid_interp!-Tuple{Any,Any,Array,Any,Any}","page":"Internal","title":"FieldTracer.grid_interp!","text":"grid_interp!(x, y, field, ix, iy)\n\nInterpolate a value at (x,y) in a field. ix and iy are indexes for x,y locations (0-based).\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-Union{Tuple{T}, Tuple{T,T,Any,Any,Any,Any}} where T<:Integer","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.normalize_field-Union{Tuple{T}, Tuple{T,T,T,Any,Any,Any,Any,Any,Any}} where T<:Integer","page":"Internal","title":"FieldTracer.normalize_field","text":"Create unit vectors of field in normalized coordinates.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.select_seeds-Tuple{Any,Any}","page":"Internal","title":"FieldTracer.select_seeds","text":" select_seeds(x, y, nSeed=100)\n\nGenerate nSeed seeding points randomly in the grid range. If you specify nSeed, use the keyword input, otherwise it will be overloaded by the 3D version seed generation.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d-NTuple{6,Any}","page":"Internal","title":"FieldTracer.trace2d","text":" trace2d(fieldx, fieldy, xstart, ystart, gridx, gridy; kwargs...)\n\n2D stream tracing in structured mesh with field in 2d array and grid in 1d array.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d-Tuple{Meshes.UnstructuredMesh,Any,Any,Any,Any}","page":"Internal","title":"FieldTracer.trace2d","text":" trace2d(mesh::UGrid, vx, vy, xstart, ystart; maxIter=1000, maxLen=1000.)\n\n2D stream tracing in unstructured quadrilateral and triangular mesh.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_eul-NTuple{6,Any}","page":"Internal","title":"FieldTracer.trace2d_eul","text":" trace2d_eul(fieldx, fieldy, xstart, ystart, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridType=\"meshgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. While this is faster than rk4, it is less accurate. Only valid for regular grid with coordinates gridx, gridy. If gridx and gridy are not given, assume that xstart and ystart are normalized coordinates (e.g. position in terms of array indices.)??? The field can be in both meshgrid (default) or ndgrid format. Supporting direction for {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace2d_rk4-NTuple{6,Any}","page":"Internal","title":"FieldTracer.trace2d_rk4","text":" trace2d_rk4(fieldx, fieldy, xstart, ystart, gridx, gridy;\n\t maxstep=20000, ds=0.01, gridType=\"meshgrid\", direction=\"both\")\n\nGiven a 2D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Runge Kutta 4. Slower than Euler, but more accurate. The higher accuracy allows for larger step sizes ds.  Only valid for regular grid with coordinates gridx, gridy. If gridx and gridy are not given, assume that xstart and ystart are normalized coordinates (e.g., position in terms of array indices.???) The field can be in both meshgrid (default) or ndgrid format. Supporting direction for {\"both\",\"forward\",\"backward\"}.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trace3d_eul-NTuple{9,Any}","page":"Internal","title":"FieldTracer.trace3d_eul","text":" trace3d_eul(fieldx, fieldy, fieldz, xstart, ystart, zstart, gridx, gridy,\n   gridz; maxstep=20000, ds=0.01)\n\nGiven a 3D vector field, trace a streamline from a given point to the edge of the vector field. The field is integrated using Euler's method. Only valid for regular grid with coordinates gridx, gridy, gridz. The field can be in both meshgrid or ndgrid (default) format.\n\n\n\n\n\n","category":"method"},{"location":"internal/#FieldTracer.trilin_reg-NTuple{4,Any}","page":"Internal","title":"FieldTracer.trilin_reg","text":"trilin_reg(x, y, z, Q)\n\nTrilinear interpolation for x1,y1,z1=(0,0,0) and x2,y2,z2=(1,1,1) Q's are surrounding points such that Q000 = F[0,0,0], Q100 = F[1,0,0], etc.\n\n\n\n\n\n","category":"method"},{"location":"log/#Log","page":"Log","title":"Log","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Streamline and trajectory are related topics in physical modeling, common seen in fluid and particle simulations.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The original version is in C from LANL. The algorithm has been reimplemented in Julia. Calling the native functions in Julia is about 5 times faster than calling the dynmic C library. Some of the tests originates from SpacePy. A bug has been fixed for the nonuniform dx, dy and dz grid.","category":"page"},{"location":"log/#Streamline-Tracing","page":"Log","title":"Streamline Tracing","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"First make it work, then make it better and fast.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"I found an approach called Pollock method.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"I need an adaptive step control integration scheme like rk45.","category":"page"},{"location":"log/#Tracing-in-Unstructured-Grid","page":"Log","title":"Tracing in Unstructured Grid","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Given an unstructured grid with node points and connectivity, how should you do the streamline tracing?","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Brute force algorithm:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"find the grid cell you are currently in;\nmove along the vector direction until you hit the boundary of that cell;\nfind the neighbour who shares the same edge and the intersection point;\nuse the vector direction in the next cell and move along that direction;\nrepeat 2-4 until you reach any of the stopping criteria: hit the boundary, exceed MaxIteration, or exceed MaxLength.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Some questions during the process:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"How to find the neighbouring cell?\nHow to determine which boundary edge will you cross?\nHow to improve the search speed?\nHow to improve accuracy?","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"A package UnstructuredGrids.jl already exists. I take advantage of this package and quickly build a 2D stream tracer on unstructured 2D grid based on the brute force algorithm. Unfortunately it is no longer maintained. I substituted it with Meshes.jl, which provides a cleaner interface and seems to be carefully designed.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Actually, the brute force algorithm may not be as bad as you think in terms of accuracy. Finite volume method uses one value per cell to represent the solution space, therefore it is just cheating to use higher order method for stream tracing.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"An example is shown for the 2D streamline tracing in the unstructured triangular mesh for the famous airfoil problem. The blue lines are the analytic stream functions derived from incompressible Euler equations which are calculated numerically. Three colored lines are displayed with dots representing the footprint inside each cell.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"An extension to 3D is possible and I may work on it in the future.","category":"page"},{"location":"log/#MATLAB","page":"Log","title":"MATLAB","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is an implementation of streamline tracing in Matlab called tristream. It requires nodal data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Inside the function, there is an intrinsic function called pointLocation, which returns the index of cell where the point locates.","category":"page"},{"location":"log/#yt","page":"Log","title":"yt","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"There is another implementation in yt library, which has many similarities to the one I borrowed from SpacePy.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Streamlining through a volume is useful for a variety of analysis tasks. By specifying a set of starting positions, the user is returned a set of 3D positions that can, in turn, be used to visualize the 3D path of the streamlines. Additionally, individual streamlines can be converted into YTStreamline objects, and queried for all the available fields along the streamline.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The implementation of streamlining in yt is described below.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"Decompose the volume into a set of non-overlapping, fully domain tiling bricks, using the AMRKDTree homogenized volume.\nFor every streamline starting position:","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"While the length of the streamline is less than the requested length:\nFind the brick that contains the current position\nIf not already present, generate vertex-centered data for the vector fields defining the streamline.\nWhile inside the brick\nIntegrate the streamline path using a Runge-Kutta 4th order method and the vertex centered data.\nDuring the intermediate steps of each RK4 step, if the position is updated to outside the current brick, interrupt the integration and locate a new brick at the intermediate position.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"The set of streamline positions are stored in the Streamlines object.","category":"page"},{"location":"log/#VTK","page":"Log","title":"VTK","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"In the VTK library, there is a class called vtkPointLocator. It is a spatial search object to quickly locate points in 3D. vtkPointLocator works by dividing a specified region of space into a regular array of \"rectangular\" buckets, and then keeping a list of points that lie in each bucket. Typical operation involves giving a position in 3D and finding the closest point. It supports both nodal data and cell data.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"vtkPointLocator has two distinct methods of interaction. In the first method, you supply it with a dataset, and it operates on the points in the dataset. In the second method, you supply it with an array of points, and the object operates on the array.","category":"page"},{"location":"log/","page":"Log","title":"Log","text":"note: Note\nMany other types of spatial locators have been developed such as octrees and kd-trees. These are often more efficient for the operations described here.","category":"page"},{"location":"log/#Limitations","page":"Log","title":"Limitations","text":"","category":"section"},{"location":"log/","page":"Log","title":"Log","text":"Currently most tracing functions in the package assume regular grid, meaning that the size of each cell is not changing. However, the whole grid information is passed all the way down to the kernel functions, which may be a waste of memory.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = FieldTracer","category":"page"},{"location":"#FieldTracer","page":"Home","title":"FieldTracer","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Vector field tracing on common types of meshes.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package supports 2nd order and 4th order field line tracing on","category":"page"},{"location":"","page":"Home","title":"Home","text":"2D/3D regular Cartesian mesh\n2D unstructured quadrilateral mesh\n2D unstructured triangular mesh","category":"page"},{"location":"example/#Examples","page":"Example","title":"Examples","text":"","category":"section"},{"location":"example/#Structured-2D-mesh","page":"Example","title":"Structured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Magnetic field line tracing","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"Streamline tracing in a 2D Earth magnetosphere simulation","category":"page"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"},{"location":"example/#Unstructured-2D-mesh","page":"Example","title":"Unstructured 2D mesh","text":"","category":"section"},{"location":"example/","page":"Example","title":"Example","text":"(Image: )","category":"page"}]
}
