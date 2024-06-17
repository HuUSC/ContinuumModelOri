from firedrake import *
from firedrake.output import VTKFile

# Create mesh and define function space
N = 30
mesh = UnitSquareMesh(N, N, diagonal='crossed')
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)

# Define boundary condition
x = SpatialCoordinate(mesh)
u1 = as_vector((x[0] + .2, x[1], 0)) #(x[1] * (1 - x[1]) * 4)
u2 = as_vector((x[0] - .2, x[1], 0))
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
a = inner(grad(grad(u)), grad(grad(v)))*dx \
  - inner(dot(avg(grad(grad(u))), n('+')), jump(grad(v)))*dS \
  - inner(jump(grad(u)), dot(avg(grad(grad(v))), n('+')))*dS \
  + alpha/h_avg*inner(jump(grad(u)), jump(grad(v)))*dS

#Penalty term for the gradient Dirichlet bc
a += alpha/h * inner(dot(grad(u), n), dot(grad(v), n)) * (ds(1) + ds(2))

# Define linear form
#f = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])
#f = Constant((0, 0, 1e-3))
#L = inner(f, v)*dx

#Rhs penalty term
G2 = Constant((1, 0, -1))
G1 = Constant((-1, 0, -1))
L = alpha/h * inner(G1, dot(grad(v), n)) * ds(1) - inner(G1, div(grad(v))) * ds(1)
L += alpha/h * inner(G2, dot(grad(v), n)) * ds(2) - inner(G2, div(grad(v))) * ds(2)

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

