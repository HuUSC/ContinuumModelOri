from dolfinx import log, default_scalar_type, mesh, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
import pyvista
from dolfinx import nls
from basix.ufl import element, mixed_element
import numpy as np
from ufl import *
from dolfinx.fem import FunctionSpace, dirichletbc, locate_dofs_topological, functionspace, Function
from mpi4py import MPI
from dolfinx.mesh import CellType, GhostMode
from ufl import (CellDiameter, FacetNormal, avg, dS, dx, grad, inner,
                 jump, pi, sin, cos)

# domain, mesh
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=((0.0, 0.0), (1.0, 1.0)), n=(5, 5),
                            cell_type=CellType.triangle,
                            ghost_mode=GhostMode.shared_facet)

Vu = element("P", domain.basix_cell(),2, shape=(3,))
Vt = element("P", domain.basix_cell(),1)
MX = functionspace(domain, mixed_element([Vu, Vt]))

# define trial and test functions
U = Function(MX) # displacement, actuation
u, theta = split(U)
V = TestFunction(MX)
w, eta = split(V)

# elasticity parameters
c_1, c_2, d_1, d_2, d_3 = 1, 1, 1E-03, 1E-03, 1E-03

theta = variable(theta)

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
e_1 = as_vector([1.0, 0.0])
e_2 = as_vector([0.0, 1.0])
phi = pi/6
u_0 = as_vector(sqrt(3) * cos(phi/2) * e_1)
v_0 = as_vector(2 * sqrt(2/(5-3 * cos(phi))) * e_2)
A_t = as_matrix([[cos((theta+phi)/2)/cos(phi/2),0],[0, sqrt((5-3 * cos(phi+theta))/(5-3 * cos(phi)))]])
u_t = A_t * u_0
v_t = A_t * v_0
I = as_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# variational form
F = variable(grad(u)) # variable grad(u)
G = variable(grad(grad(u))) # variable grad(grad(u))
n = cross(F*e_1,F*e_2)/sqrt(inner(cross(F*e_1,F*e_2) , cross(F*e_1,F*e_2)))
L = F.T * F - A_t.T * A_t
q = dot(diff(v_t,theta) , v_t) * inner(G,outer(n,u_0,u_0)) + dot(diff(u_t,theta) , u_t) * inner(G,outer(n,v_0,v_0))
P = (
      2 * c_1 * F * (L + L.T)
      + 2 * c_2 * q * dot( diff(v_t,theta) , v_t) / sqrt(inner(cross(F*e_1,F*e_2) , cross(F*e_1,F*e_2))) * (
      outer( cross( F*e_2, ( I - outer(n,n) ) * dot( dot(G,u_0), u_0 ) ) , e_1)
      - outer( cross( F*e_1, ( I - outer(n,n) )* dot( dot(G,u_0), u_0) ) , e_2) )
      + 2 * c_2 * q * dot(diff(u_t,theta) , u_t) / sqrt(inner(cross(F*e_1,F*e_2) , cross(F*e_1,F*e_2))) * (
              outer(cross(F * e_2, (I - outer(n, n) ) * dot(dot(G, v_0), v_0)), e_1)
              - outer(cross(F * e_1, (I - outer(n, n)) * dot( dot(G, v_0), v_0)), e_2) )
      )
H = ( 2 * c_2 * q * ( dot( diff(v_t,theta) , v_t) * outer(n,u_0,u_0) + dot( diff(u_t,theta) , u_t) * outer(n,v_0,v_0) )
    + 2 * d_3 * G )
f = - 2 * c_1 * inner( L , diff( A_t.T * A_t, theta ) ) + 2 * c_2 * q * diff( q, theta ) + 2 * d_1 * theta
g = 2 * d_2 * grad(theta)

# energy density
Ee = inner( P, grad(w) ) + inner( H, grad(grad(w)) ) + inner( g, grad(eta) ) + f * eta
# interior penalty terms
alpha = default_scalar_type(1.0) # penalty parameter
h = CellDiameter(domain) # cell diameter
n_F = FacetNormal(domain) # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet
IPT = ( alpha / h_avg * inner(jump( dot( H, n_F), n_F), jump(grad(w), n_F)) - inner( avg( dot( dot( H, n_F), n_F) ) , jump( grad(w), n_F) ) -
     inner( avg( dot( grad(w), n_F ) ) , jump( dot( H, n_F), n_F) ) )
# weak form
WF = Ee * dx + IPT * dS

# boundary conditions
fdim = domain.topology.dim - 1


def clamped_top_expression(x):
    return np.stack((np.ones(x.shape[1]) * 0.5, np.ones(x.shape[1]) * 0.9, np.zeros(x.shape[1])))

def clamped_bottom_expression(x):
    return np.stack((np.ones(x.shape[1]) * 0.5, np.ones(x.shape[1]) * (0.1), np.zeros(x.shape[1])))

# fixed at x = 0.5, y = 1
def top_boundary(x):
    return np.logical_and(np.isclose( x[0],0.5), np.isclose( x[1], 1.0) )

boundary_facets_top = mesh.locate_entities_boundary(domain, fdim, top_boundary)
V1, _ = MX.sub(0).collapse()
u_top = Function(V1)
# u_top.x.array[0] = 0.5
# u_top.x.array[1] = 1.1
# u_top.x.array[2] = 0.0
u_top.interpolate(clamped_top_expression)
bc_top = dirichletbc(u_top, locate_dofs_topological((MX.sub(0), V1), fdim, boundary_facets_top))

# fixed at x = 0.5, y = 0
def bottom_boundary(x):
     return np.logical_and(np.isclose( x[0],0.5), np.isclose( x[1], 0.0) )

boundary_facets_bottom = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
u_bottom = Function(V1)
# u_bottom.x.array[0] = 0.5
# u_bottom.x.array[1] =  -0.1
# u_bottom.x.array[2] = 0.0
u_bottom.interpolate(clamped_bottom_expression)
bc_bottom = dirichletbc(u_bottom, locate_dofs_topological((MX.sub(0), V1), fdim, boundary_facets_bottom))


# Collect Dirichlet boundary conditions
bcs = [bc_top, bc_bottom]

# solve the problem
problem = NonlinearProblem(WF, U, bcs)

solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1E-03
solver.rtol = 1E-03
solver.max_it = 100
solver.convergence_criterion = "incremental"
solver.report = True

log.set_log_level(log.LogLevel.INFO)
N, converged = solver.solve(U)
assert (converged)
print(f"Number of interations: {N:d}")

uh, theta_h = U.split()

with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as file:
    U1 = functionspace(domain, element("P", domain.basix_cell(),1, shape=(3,)))
    u1 = Function(U1)
    u1.interpolate(uh)
    file.write_mesh(domain)
    file.write_function(u1)
