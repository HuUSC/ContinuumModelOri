from firedrake import *
import sys

# Assume mesh and V are already defined
mesh = UnitSquareMesh(10, 10, name="meshA")
V = FunctionSpace(mesh, "CG", 1)

# Perform some computation
result = Function(V, name='res')
result.interpolate(Constant(1))

# Saving the result
with CheckpointFile("example.h5", 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(result)

# Load the result
with CheckpointFile("example.h5", 'r') as afile:
    mesh = afile.load_mesh("meshA")
    f = afile.load_function(mesh, "res")

