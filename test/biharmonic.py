from firedrake import *
from firedrake.output import VTKFile

# Create mesh and define function space
N = 10
mesh = UnitSquareMesh(N, N, diagonal='crossed')
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)

# Define boundary condition
u1 = Constant((.15, 0, 0))
u2 = Constant((-.15, 0, 0))
bcs = [DirichletBC(V, u1, 1), DirichletBC(V, u2, 2)]

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
  + alpha/h_avg*inner(jump(grad(u)), jump(grad(v)))*dS

#Penalty term for the gradient Dirichlet bc
a += alpha/h * inner(dot(grad(u), n), dot(grad(v), n)) * (ds(1) + ds(2))

# Define linear form
x = SpatialCoordinate(mesh)
#f = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])
f = Constant((0, 0, 1e-3))
L = inner(f, v)*dx

#Rhs penalty term
Gd = Constant((-1, 0, -1))
L = alpha/h * inner(Gd, dot(grad(v), n)) * (ds(1) + ds(2))

# Solve variational problem
u = Function(V, name='sol')
solve(a == L, u, bcs)

# Save solution to file
file = VTKFile("sol.pvd")
file.write(u)

file = VTKFile("3d.pvd")
aux = Function(V, name='sol 3d')
aux.interpolate(u - as_vector((x[0], x[1], 0)))
file.write(aux)

