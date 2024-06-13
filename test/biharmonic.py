from firedrake import *
from firedrake.output import VTKFile

# Create mesh and define function space
N = 10
mesh = UnitSquareMesh(N, N, diagonal='crossed')
V = FunctionSpace(mesh, "CG", 2)

# Define boundary condition
u0 = Constant(0.0)
bcs = [DirichletBC(V, u0, 1), DirichletBC(V, u0, 2), DirichletBC(V, u0, 3), DirichletBC(V, u0, 4)]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)

# Penalty parameter
alpha = Constant(8.0)

# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
x = SpatialCoordinate(mesh)
f = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])
L = f*v*dx

# Solve variational problem
u = Function(V, name='sol')
solve(a == L, u, bcs)

# Save solution to file
file = VTKFile("biharmonic.pvd")
file.write(u)

