# Log

Original version in C from LANL. Reimplement in Julia.
Calling the native functions in Julia is about 5 times faster than calling the
dynmic C library.
Some of the tests originates from [SpacePy](https://github.com/spacepy/spacepy).
Bug fixed for nonuniform `dx`, `dy` and `dz` grid.

## Limitations

* Currently most tracing functions assume regular grid, meaning that the size of each cell is not changing. However, the whole grid information is passed all the way down to the kernel functions, which may be a waste of memory.