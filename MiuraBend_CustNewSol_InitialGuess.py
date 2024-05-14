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
NN = 20
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=( (0.0, 0.0), (a, b) ), n=(NN, NN),
                            cell_type=CellType.triangle,
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
Q0 = MX.sub(0).sub(1) # subspace for x-component of displacement
Qx, _ = Q0.collapse()
Q2 = MX.sub(0).sub(2) # subspace for z-component of displacement
Qz, _ = Q2.collapse()
# V1 = MX.sub(1)
# R, _ = V1.collapse()

# boundary conditions of displacement
def boundary_zero(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))
#
# def boundary_right(x):
#     return np.zeros_like(x[0])
#
# def boundary_left(x):
#     return np.zeros_like(x[0])
#
# def boundary_top(x):
#     return np.vstack( ( np.zeros_like(x[0]), np.ones_like(x[0]) * (-0.1), np.zeros_like(x[0]) ))
#
# def boundary_bottom(x):
#     return np.vstack( ( np.zeros_like(x[0]), np.ones_like(x[0]) * (0.1), np.zeros_like(x[0]) ))

def boundary_CL(x):
    return np.vstack( ( np.ones_like(x[0]) * (0.1), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CR(x):
    return np.vstack( ( np.ones_like(x[0]) * (-0.1), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_center(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]) * (0.05) ))

fdim = domain.topology.dim - 1
#
# facets_R = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], a) )
# dofs_R = locate_dofs_topological((Q0, Qx), 1, facets_R)
# u_R = Function(Qx)
# u_R.interpolate(boundary_right)
# bc_R = dirichletbc(u_R, dofs_R, V0)
#
# facets_L = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], 0.0) )
# dofs_L = locate_dofs_topological((Q0, Qx), 1, facets_L)
# u_L = Function(Qx)
# u_L.interpolate(boundary_left)
# bc_L = dirichletbc(u_L, dofs_L, V0)
#
# facets_T = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], 0.0), np.isclose( x[1], b)) )
# dofs_T = locate_dofs_topological((V0, Q), 0, facets_T)
# u_T= Function(Q)
# u_T.interpolate(boundary_top)
# bc_T = dirichletbc(u_T, dofs_T, V0)
#
# facets_B = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a), np.isclose( x[1],0.0)) )
# dofs_B = locate_dofs_topological((V0, Q), 0, facets_B)
# u_B = Function(Q)
# u_B.interpolate(boundary_bottom)
# bc_B = dirichletbc(u_B, dofs_B, V0)

# facets_CT = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1], b)) )
# dofs_CT = locate_dofs_topological((Q2, Qz), 0, facets_CT)
# u_CT = Function(Qz)
# u_CT.interpolate(boundary_top_C)
# bc_CT = dirichletbc(u_CT, dofs_CT, Q2)
#
# facets_CB = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1], 0.0)) )
# dofs_CB = locate_dofs_topological((Q2, Qz), 0, facets_CB)
# u_CB = Function(Qz)
# u_CB.interpolate(boundary_top_C)
# bc_CB = dirichletbc(u_CB, dofs_CB, Q2)

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
bcs = [bc_CR, bc_CL, bc_C]
# bcs = [bc_T, bc_B, bc_R, bc_L, bc_CR, bc_CL]

# Create tags for facets subjected to gradient Dirichlet boundary conditions
marked_facets = np.hstack([facets_CL, facets_CR])
marked_values = np.hstack([np.full_like(facets_CL, 1), np.full_like(facets_CR, 1)])
# marked_facets = facets_CL
# marked_values = np.full_like(facets_CL, 1)
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

# elasticity parameters
c_1, c_2, d_1, d_2, d_3 = 12.0, 12.0, 1E-2, 1E-1, 1E-2
# c_1, c_2, d_1, d_2, d_3 = 10.0, 10.0, 1E-3, 1E-3, 1E-3 diverge

# theta = variable(theta)

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
# phi = pi / 6
# u_s = sqrt(3.0) * cos(phi / 2)
# v_s = 2 * sqrt(2.0 / (5 - 3 * cos(phi)))
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


# variational form
F = I + grad(u)  # variable grad(u)
G = variable( grad(grad(u)) )  # variable grad(grad(u))
n = cross( F*e_1, F * e_2 ) / sqrt( inner( cross( F*e_1 , F*e_2 ), cross( F*e_1, F*e_2 ) ) )
# n = cross( F[:,0], F[:,1] ) / sqrt( inner( cross( F[:,0], F[:,1] ), cross( F[:,0], F[:,1] )  ) )
L = F.T * F - A_t.T * A_t
q = v_t_p * v_ts * inner( G, outer(n,u_0,u_0)  ) \
    + u_t_p * u_ts * inner( G,outer(n,v_0,v_0) )
# H =  2 * c_2 * q * ( inner( v_t_p , v_t) * outer(n,u_0,u_0) + inner( u_t_p , u_t) * outer(n,v_0,v_0) ) \
#      + 2 * d_3 * G


# energy density function
W = c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( G, G )
H = diff( W, G )
# Total potential energy
Ee = W * dx

# interior penalty terms
alpha = default_scalar_type(0.1) # penalty parameter
h = CellDiameter(domain) # cell diameter
n_F = FacetNormal(domain) # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet
IPT = 1.0/2.0 * alpha / h_avg * inner( jump( F, n_F ), jump( F, n_F ) ) \
        - inner( avg( dot( dot( H, n_F), n_F) ) , jump( F, n_F) )

# weak gradient Dirichlet boundary conditions
GDBC = 1.0/2.0 * alpha / h * inner( dot( grad(u) , n_F),  dot( grad(u) , n_F) ) \
       - inner(  dot( grad(u) , n_F), dot( dot( H, n_F), n_F) )

# first variation of Ee
Ee_Var = Ee + IPT * dS + GDBC * ds(1)
# Ee_Var = Ee + IPT * dS

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

# ########### initial guess from linear problem
UL = TrialFunction(MX)
VL = TestFunction(MX)
ul, thetal = split(UL)
wl, etal = split(VL)

f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
WL = 2 * thetal * etal + 2 * inner(grad(thetal), grad(etal)) + 2 * inner( grad(grad(ul)), grad(grad(wl)) )
IPTL =  alpha / h_avg * inner( jump( grad(ul), n_F ), jump( grad(wl), n_F ) ) \
        - inner( avg( dot( dot( grad(grad(ul)), n_F), n_F) ) , jump( grad(wl), n_F) ) - \
     inner( avg( dot( grad(wl), n_F ) ) , jump( dot( grad(grad(ul)), n_F), n_F) )
aL = WL * dx + IPTL * dS
LL = inner(f, wl) * dx
problemL = LinearProblem(aL, LL, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
U0 = problemL.solve()

# uhl, theta_hl = U0.sub(0).collapse(), U0.sub(1).collapse()
#
# Vu2 = element("P", domain.basix_cell(),1, shape=(3,))
# V2 = functionspace(domain, Vu2)
# uhl_i = Function(V2)
# uhl_i.interpolate(uhl)
#
# # Create plotter and pyvista grid
# pl = pyvista.Plotter()
# topology, cell_types, geometry = plot.vtk_mesh(V2)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#
# # Attach vector values to grid and warp grid by vector
# grid["ul"] = uhl_i.x.array.reshape((geometry.shape[0], 3))
# actor_0 = pl.add_mesh(grid, style="wireframe", color="k")
# warped = grid.warp_by_vector("ul", factor=1.5)
# actor_1 = p.add_mesh(warped, show_edges=True)
# pl.show_axes()
# if not pyvista.OFF_SCREEN:
#     pl.show()
# else:
#     figure_as_array = pl.screenshot("displacement.png")
######################

max_iterations = 100

i = 0
dU_norm = []
# dU = Function(MX)
# initial guess
dU = U0
while i < max_iterations:
    # Update u_{i+1} = u_i + delta u_i
    U.x.array[:] += dU.x.array

    # Compute norm of update
    correction_norm = dU.vector.norm(0)
    dU_norm.append(correction_norm)

    if correction_norm < 1e-8:
        break

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
    i += 1

# # Convergence rate
# fig = plt.figure(figsize=(15, 8))
fig = plt.figure()
plt.title(r"Residual of $\vert\vert\delta u_i\vert\vert$")
plt.plot(np.arange(i+1), dU_norm)
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
Vu2 = element("P", domain.basix_cell(),1, shape=(3,))
V2 = functionspace(domain, Vu2)
uh_i = Function(V2)
uh_i.interpolate(uh)

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh_i.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
    p.screenshot("Miura_Bend_PointDisplacement_Clamped.png")
else:
    figure_as_array = p.screenshot("deflection.png")

# # Export displacement field
# num_sub_spaces = 3
# num_dofs_per_component = int(len(uh_i.x.array)/3)
# # print(num_dofs_per_component, num_sub_spaces)
# vector = np.zeros((num_sub_spaces, num_dofs_per_component))
# for i in range(num_sub_spaces):
#     vector[i] = uh_i.sub(i).collapse().x.array
# xx = V2.tabulate_dof_coordinates()
# vector = vector.T
# out = open("Displacement_SS.csv", "w")
# for coord, vec in zip(xx, vector):
#     print(f"{coord[0]}, {coord[1]}, {coord[0]}, {coord[1]}, {vec[2]}", file=out)
# out.close()