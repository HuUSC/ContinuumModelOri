from dolfinx import log, default_scalar_type, mesh, io, plot, default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
import pyvista
from dolfinx import nls
from basix.ufl import element, mixed_element
import numpy as np
from ufl import *
from dolfinx.fem import FunctionSpace, dirichletbc, locate_dofs_topological, functionspace, Function, form
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import CellType, GhostMode
from ufl import (CellDiameter, FacetNormal, avg, dS, dx, grad, inner,
                 jump, pi, sin, cos, as_vector, as_matrix)
import dolfinx.fem.petsc

# domain, mesh
phi = pi / 6
u_s = sqrt(3.0) * cos(phi / 2)
v_s = 2 * sqrt(2.0 / (5 - 3 * cos(phi)))
a, b = u_s/v_s , 1.0
NN = 30
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=( (0.0, 0.0), (a, b) ), n=(NN, NN),
                            cell_type=CellType.quadrilateral,
                            ghost_mode=GhostMode.shared_facet)

Vu = element("P", domain.basix_cell(),2, shape=(3,))
Vt = element("P", domain.basix_cell(),1)
MX = functionspace(domain, mixed_element([Vu, Vt]))


# define trial and test functions
U = Function(MX)
u, theta = split(U) # displacement, actuation field
V = TestFunction(MX)
w, eta = split(V)


# get subspace of MX
V0 = MX.sub(0)
Q, _ = V0.collapse()
# Q0 = MX.sub(0).sub(1) # subspace for x-component of displacement
# Qx, _ = Q0.collapse()
# Q2 = MX.sub(0).sub(2) # subspace for z-component of displacement
# Qz, _ = Q2.collapse()
# V1 = MX.sub(1)
# R, _ = V1.collapse()

# boundary conditions of displacement
def boundary_zero(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CL(x):
    return np.vstack( ( np.ones_like(x[0]) * (0.15), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CR(x):
    return np.vstack( ( np.ones_like(x[0]) * (-0.15), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_center(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]) * (0.05) ))

# gradient Dirichlet boundary conditions
def GR_b(x):
    return np.vstack( ( np.ones_like(x[0])*(1.0), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0) ) )

def GL_b(x):
    return np.vstack( ( np.ones_like(x[0])*(-1.0), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0) ) )


# ## Form displacement Dirichlet boundary conditions
fdim = domain.topology.dim - 1

facets_CR = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], a), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
dofs_CR = locate_dofs_topological((V0, Q), 1, facets_CR)
u_CR = Function(Q)
u_CR.interpolate(boundary_CR)
bc_CR = dirichletbc(u_CR, dofs_CR, V0)

facets_CL = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], 0.0), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
dofs_CL = locate_dofs_topological((V0, Q), 1, facets_CL)
u_CL = Function(Q)
u_CL.interpolate(boundary_CL)
bc_CL = dirichletbc(u_CL, dofs_CL, V0)

facets_whole = mesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or.reduce((np.isclose(x[0],0.0),
                    np.isclose(x[0], a), np.isclose(x[1], 0.0), np.isclose(x[1], b))))
dofs_whole = locate_dofs_topological((V0, Q), fdim, facets_whole)
u_whole= Function(Q)
u_whole.interpolate(boundary_zero)
bc_whole = dirichletbc(u_whole, dofs_whole, V0)

facets_C = mesh.locate_entities( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1], b/2)) )
dofs_C = locate_dofs_topological((V0, Q), 0, facets_C)
u_C = Function(Q)
u_C.interpolate(boundary_center)
bc_C = dirichletbc(u_C, dofs_C, V0)

# Collect displacement Dirichlet boundary conditions
bcs = [bc_CR, bc_CL]
# bcs = [bc_T, bc_B, bc_R, bc_L, bc_CR, bc_CL]

# Create tags for facets subjected to gradient Dirichlet boundary conditions
marked_facets = np.hstack([facets_CL, facets_CR])
marked_values = np.hstack([np.full_like(facets_CL, 1), np.full_like(facets_CR, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

# ## kinematics of design
u_ts = sqrt(3.0) * cos((theta + phi) / 2)
v_ts = 2 * sqrt(2.0 / (5 - 3 * cos(theta + phi)))
e_1 = as_vector([1.0, 0.0])
e_2 = as_vector([0.0, 1.0])
u_0 = u_s * e_1
v_0 = v_s * e_2
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, variable(theta))
v_t_p = diff(v_ts, variable(theta))
I = as_matrix( [ [1.0, 0.0], [0.0, 1.0], [0.0, 0.0] ] )
# mesh info.
h = CellDiameter(domain) # cell diameter
n_F = FacetNormal(domain) # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet

# kinematic variables
F = I + grad(u)  # variable grad(u)
G = variable( grad(grad(u)) )  # variable grad(grad(u))
n = cross( F*e_1, F * e_2 ) / sqrt( inner( cross( F*e_1 , F*e_2 ), cross( F*e_1, F*e_2 ) ) )
L = F.T * F - A_t.T * A_t
q = v_t_p * v_ts * inner( G, outer(n,u_0,u_0)  ) \
    + u_t_p * u_ts * inner( G,outer(n,v_0,v_0) )

# elasticity parameters
c_1, c_2, d_1, d_2, d_3 = 1.0, 1.0, 1E-2, 1E-1, 1E-2


# ######## Nonlinear Plate Model

# energy density function
W = c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( G, G )
H = diff( W, G ) # third-order stress tensor
# Total potential energy
Ee = W * dx

# interior penalty terms
alpha = default_scalar_type(0.1) # penalty parameter
IPT = 1.0/2.0 * alpha / h_avg * inner( jump( F, n_F ), jump( F, n_F ) ) \
        - inner( avg( dot( dot( H, n_F), n_F) ) , jump( F, n_F) )


# # Weak form of gradient Dirichlet boundary conditions
GR_be = Function(Q)
GR_be.interpolate(GR_b)
GL_be = Function(Q)
GL_be.interpolate(GL_b)

GDBC_R = 1.0/2.0 * alpha / h * inner( dot( F , n_F) - GR_be,  dot( F , n_F) - GR_be ) \
       - inner(  dot( F, n_F) - GR_be , dot( dot( H, n_F), n_F) )

GDBC_L = 1.0/2.0 * alpha / h * inner( dot( F , n_F) - GL_be,  dot( F , n_F) - GL_be ) \
       - inner(  dot( F, n_F) - GL_be , dot( dot( H, n_F), n_F) )

# first variation of Ee
Ee_Var = Ee + IPT * dS + GDBC_R * ds(2) + GDBC_L * ds(1)

# weak form
# f = as_vector([0.0, 0.0, -0.5])
WF = derivative(Ee_Var, U, V)
J = derivative(WF, U) # Jacobian
residual = form(WF)
jacobian = form(J)

# incremental
A = fem.petsc.create_matrix(jacobian)
L = fem.petsc.create_vector(residual)
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)

max_iterations = 100

i = 0
dU_norm = []
dU = Function(MX)
# initial guess

while i < max_iterations:
    # Assemble Jacobian and residual
    with L.localForm() as loc_L:
        loc_L.set(0)
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, jacobian, bcs=bcs)
    A.assemble()
    dolfinx.fem.petsc.assemble_vector(L, residual)
    L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    L.scale(-1)

    # Compute b - J(u_D-u_(i-1))
    dolfinx.fem.petsc.apply_lifting(L, [jacobian], [bcs], x0=[U.vector], scale=1)
    # Set du|_bc = u_{i-1}-u_D
    dolfinx.fem.petsc.set_bc(L, bcs, U.vector, 1.0)
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(L, dU.vector)
    dU.x.scatter_forward()

    # Update u_{i+1} = u_i + delta u_i
    U.x.array[:] += dU.x.array
    i += 1

    # Compute norm of update
    correction_norm = dU.vector.norm(0)
    dU_norm.append(correction_norm)

    if correction_norm < 1e-8:
        break


# # Convergence rate
# fig = plt.figure(figsize=(15, 8))
fig = plt.figure()
plt.title(r"Residual of $\vert\vert\delta u_i\vert\vert$")
plt.plot(np.arange(i), dU_norm)
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel("Iterations")
plt.ylabel(r"$\vert\vert \delta u\vert\vert$")
plt.grid()
plt.show()

# split the solution
uh, theta_h = U.sub(0).collapse(), U.sub(1).collapse()
# print(uh.x.array)

## plot the deformation
VT = functionspace(domain, element("P", domain.basix_cell(),1) )
Vu2 = element("P", domain.basix_cell(),1, shape=(3,))
V2 = functionspace(domain, Vu2)
uh_i = Function(V2)
uh_i.interpolate(uh)
# U0_i = Function(V2)
# U0_i.interpolate(U0)

# Create plotter and pyvista grid
# p = pyvista.Plotter()
# topology, cell_types, geometry = plot.vtk_mesh(V2)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#
# # Attach vector values to grid and warp grid by vector
# grid["u"] = uh_i.x.array.reshape((geometry.shape[0], 3))
# actor_0 = p.add_mesh(grid, style="wireframe", color="k")
# warped = grid.warp_by_vector("u", factor=1.5)
# actor_1 = p.add_mesh(warped, show_edges=True)
# p.show_axes()
# if not pyvista.OFF_SCREEN:
#     p.show()
# else:
#     figure_as_array = p.screenshot("deflection.png")

p = pyvista.Plotter(shape=(1, 2))


p.subplot(0, 0)  # plot displacement field
topology, cell_types, geometry = plot.vtk_mesh(V2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = uh_i.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()


p.subplot(0, 1)  # plot actuation field
topology_h, cell_types_h, geometry_h = plot.vtk_mesh(VT)
grid_h = pyvista.UnstructuredGrid(topology_h, cell_types_h, geometry_h)
grid_h.point_data["th"] = theta_h.x.array
warped_h = grid_h.warp_by_scalar("th", factor=25)
actor_h = p.add_mesh(warped_h, show_edges=True, show_scalar_bar=True, scalars="th")
p.show_axes()

p.show()
p.screenshot("Miura_Field_Zero.png")