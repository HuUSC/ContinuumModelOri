from dolfinx import log, default_scalar_type, mesh, io, plot, default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
import pyvista
from dolfinx import nls
from basix.ufl import element, mixed_element
import numpy as np
from ufl import *
from dolfinx.fem import FunctionSpace, dirichletbc, locate_dofs_topological, functionspace, Function, form, Expression
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import CellType, GhostMode
from ufl import (CellDiameter, FacetNormal, avg, dS, dx, grad, inner,
                 jump, pi, sin, cos, as_vector, as_matrix)
import dolfinx.fem.petsc
import vtk
vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()

# domain, mesh
phi = pi / 6.0
u_s = sqrt(3.0) * cos(phi / 2.0)
v_s = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(phi)))
a, b = u_s/v_s, 1.0
NN = 30
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=( (0.0, 0.0), (a, b) ), n=(NN, NN),
                            cell_type=CellType.quadrilateral,
                            ghost_mode=GhostMode.shared_facet)

Vu = element("P", domain.basix_cell(),2, shape=(3,))
Vt = element("P", domain.basix_cell(),1)
MX = functionspace(domain, mixed_element([Vu, Vt]))

# get subspace of MX
V0 = MX.sub(0)
Q, Q_map = V0.collapse()
# Q0 = MX.sub(0).sub(1) # subspace for x-component of displacement
# Qx, _ = Q0.collapse()
# Q2 = MX.sub(0).sub(2) # subspace for z-component of displacement
# Qz, _ = Q2.collapse()
V1 = MX.sub(1)
R, R_map = V1.collapse()

# boundary conditions of displacement
def boundary_zero(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

# def boundary_CL(x):
#     return np.vstack( ( x[0] + np.ones_like(x[0]) * (0.1), x[1] + np.zeros_like(x[0]), (x[1]-b/2)**2 ))
#
# def boundary_CR(x):
#     return np.vstack( ( x[0] + np.ones_like(x[0]) * (-0.1), x[1] + np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CL(x):
    return np.vstack( ( x[0] + np.ones_like(x[0]) * (0.2), x[1], np.zeros_like(x[0]) ))

def boundary_CR(x):
    return np.vstack( ( x[0] + np.ones_like(x[0]) * (-0.2), x[1], np.zeros_like(x[0]) ))

def boundary_center(x):
    return np.vstack( ( x[0], x[1], np.zeros_like(x[0]) ))

# gradient Dirichlet boundary conditions
def GR_b(x):
    return np.vstack( ( np.ones_like(x[0])*(1.0/sqrt(2)), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0/sqrt(2)) ) )

def GL_b(x):
    return np.vstack( ( np.ones_like(x[0])*(-1.0/sqrt(2)), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0/sqrt(2)) ) )


slope = 's'  # empty if there is no gradient Dirichlet boundary condition

# ## Form displacement Dirichlet boundary conditions
fdim = domain.topology.dim - 1

facets_CR = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], a), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
# facets_CR = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], a) )
dofs_CR = locate_dofs_topological((V0, Q), 1, facets_CR)
u_CR = Function(Q)
u_CR.interpolate(boundary_CR)
bc_CR = dirichletbc(u_CR, dofs_CR, V0)

facets_CL = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], 0.0), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
# facets_CL = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], 0.0) )
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

# elasticity parameters
c_1, c_2, d_1, d_2, d_3 = 1.0, 1.0, 1e-2, 1e-2, 1e-2
I = as_matrix( [ [1.0, 0.0], [0.0, 1.0], [0.0, 0.0] ] )
# mesh info.
h = CellDiameter(domain) # cell diameter
n_F = FacetNormal(domain) # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet

# ####### initial guess for displacement field from 3D bi-Hessian
VL = functionspace(domain, element("P", domain.basix_cell(),2, shape=(3,)))
# VL = Q
ul = TrialFunction(VL)
wl = TestFunction(VL)

dofs_CR_l = locate_dofs_topological(VL, 1, facets_CR)
ul_CR = Function(VL)
ul_CR.interpolate(boundary_CR)
bc_CR_l = dirichletbc(ul_CR, dofs_CR_l)

dofs_CL_l = locate_dofs_topological(VL, 1, facets_CL)
ul_CL = Function(VL)
ul_CL.interpolate(boundary_CL)
bc_CL_l = dirichletbc(ul_CL, dofs_CL_l)

dofs_C_l = locate_dofs_topological(VL, 0, facets_C)
ul_C = Function(VL)
ul_C.interpolate(boundary_center)
bc_C_l = dirichletbc(ul_C, dofs_C_l)

bcs_l = [bc_CR_l, bc_CL_l]

alpha_l, alpha_d = 20.0, 20.0

IPTL = alpha_l / h_avg * inner( jump( grad(ul) ), jump( grad(wl) ) ) \
       - inner( avg( dot( dot( grad(grad(ul)), n_F), n_F) ) , jump( grad(wl), n_F) )\
       - inner( avg( dot( dot( grad(grad(wl)), n_F), n_F) ) , jump( grad(ul), n_F) )

GR_be_l = Function(VL)
GR_be_l.interpolate(GR_b)
GL_be_l = Function(VL)
GL_be_l.interpolate(GL_b)

GDBC_RL_bi = alpha_d / h * inner( dot( grad(ul), n_F), dot( grad(wl), n_F) ) \
             - inner( dot( grad(wl), n_F), dot( dot( grad(grad(ul)), n_F), n_F) )\
             - inner( dot( grad(ul), n_F), dot( dot( grad(grad(wl)), n_F), n_F) )

GDBC_RL_li = alpha_d / h * inner( GR_be_l, dot( grad(wl), n_F) ) \
             - inner( GR_be_l, dot( dot( grad(grad(wl)), n_F), n_F) )

GDBC_LL_bi = alpha_d / h * inner( dot( grad(ul), n_F), dot( grad(wl), n_F) ) \
             - inner( dot( grad(wl), n_F), dot( dot( grad(grad(ul)), n_F), n_F) )\
             - inner( dot( grad(ul), n_F), dot( dot( grad(grad(wl)), n_F), n_F) )

GDBC_LL_li = alpha_d / h * inner( GL_be_l, dot(grad(wl), n_F)) \
             - inner( GL_be_l, dot( dot( grad(grad(wl)), n_F), n_F) )

# a = inner( grad(grad(ul)), grad(grad(wl)) ) * dx + IPTL * dS + GDBC_RL_bi * ds(2) + GDBC_LL_bi * ds(1)
# L = GDBC_RL_li * ds(2) + GDBC_LL_li * ds(1)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
aa = inner( grad(grad(ul)), grad(grad(wl)) ) * dx + IPTL * dS
LL = inner(f, wl) * dx

if bool(slope):
    aa += GDBC_RL_bi * ds(2) + GDBC_LL_bi * ds(1)
    LL += GDBC_RL_li * ds(2) + GDBC_LL_li * ds(1)

problem = LinearProblem(aa, LL, bcs=bcs_l, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
U0 = problem.solve()

# ####### initial guess for angle field from the metric constraint
VT = functionspace(domain, element("P", domain.basix_cell(),1) )
# VT = R
theta_t = Function(VT)
eta_t = TestFunction(VT)

F0 = grad(U0)
u_ts_t = sqrt(3.0) * cos((variable(theta_t) + phi) / 2)
v_ts_t = 2 * sqrt(2.0 / (5 - 3 * cos(variable(theta_t) + phi)))
A_t_t = as_matrix( [ [ u_ts_t/ u_s, 0], [0, v_ts_t/v_s] ] )
L_t = F0.T * F0 - A_t_t.T * A_t_t

c_1_l, d_2_l = c_1, d_2
# c_1_l, d_2_l = 1.0, 10
WT = ( c_1_l * inner( L_t, L_t ) + d_2_l * inner( grad(theta_t), grad(theta_t) ) ) * dx
# WT = ( inner( L_t, L_t ) + inner( grad(theta_t), grad(theta_t) ) ) * dx
E_T = derivative( WT, theta_t, eta_t)
residual_T = form(E_T)
J_T = derivative(E_T, theta_t)
jacobian_T = form(J_T)

A_T = dolfinx.fem.petsc.create_matrix(jacobian_T)
L_T = dolfinx.fem.petsc.create_vector(residual_T)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A_T)
du = dolfinx.fem.Function(VT)

max_iterations = 25
i = 0
while i < max_iterations:
    # Assemble Jacobian and residual
    with L_T.localForm() as loc_L:
        loc_L.set(0)
    A_T.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A_T, jacobian_T)
    A_T.assemble()
    dolfinx.fem.petsc.assemble_vector(L_T, residual_T)
    L_T.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Scale residual by -1
    L_T.scale(-1)
    L_T.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(L_T, du.vector)
    du.x.scatter_forward()
    # Update u_{i+1} = u_i + delta u_i
    theta_t.x.array[:] += du.x.array
    i += 1

    # Compute norm of update
    correction_norm_t = du.vector.norm(0)
    # print(f"Iteration {i}: Correction norm {correction_norm_t}")
    if correction_norm_t < 1e-10:
        break


# ######## Nonlinear Plate Model
U = Function(MX)
# initial guess
U0_h = Function(Q)
U0_h.interpolate(U0)
theta_t_h = Function(R)
theta_t_h.interpolate(theta_t)
U.x.array[Q_map] = U0_h.x.array
U.x.array[R_map] = theta_t_h.x.array
U.x.scatter_forward()

# define trial and test functions
u, theta = split(U)  # displacement, actuation field
V = TestFunction(MX)
w, eta = split(V)

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

# kinematic variables
F = grad(u)  # variable grad(u)
G = variable( grad(grad(u)) )  # variable grad(grad(u))
n = cross( F*e_1, F * e_2 ) / sqrt( inner( cross( F*e_1 , F*e_2 ), cross( F*e_1, F*e_2 ) ) )
L = F.T * F - A_t.T * A_t
q = v_t_p * v_ts * inner( G, outer(n,u_0,u_0)  ) \
    + u_t_p * u_ts * inner( G,outer(n,v_0,v_0) )

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
# Ee_Var = Ee + IPT * dS + GDBC_R * ds(2) + GDBC_L * ds(1)
Ee_Var = Ee + IPT * dS

if bool(slope):
    Ee_Var += GDBC_R * ds(2) + GDBC_L * ds(1)

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

    print(f"Iteration {i}: Correction norm {correction_norm}")

    if correction_norm < 1e-10:
        break

# # Convergence rate
# fig = plt.figure(figsize=(15, 8))
fig = plt.figure()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
})
plt.plot(np.arange(i), np.log10(dU_norm))
ax = plt.gca()
# ax.set_yscale('log')
ax.set_xlabel(r"Iteration", fontsize=13)
ax.set_ylabel(r"Residual norm in log scale", fontsize=13)
ax.set_xticks(np.arange(0, i, step=1))
# ax.set_title(r"Residual")
plt.grid(linestyle='--')
# plt.savefig("Newton_Convergence.eps", dpi=300.0)
plt.show()

# split the solution
uh, theta_h = U.sub(0).collapse(), U.sub(1).collapse()
# print(uh.x.array)

# ## plot the deformation

def reference(x):
    return np.vstack( ( x[0], x[1], np.zeros_like(x[0]) ))


Vu2 = element("P", domain.basix_cell(),1, shape=(3,))
V2 = functionspace(domain, Vu2)
uh_i = Function(V2)
uh_i.interpolate(uh)
U0_i = Function(V2)
U0_i.interpolate(U0)

r_i = Function(V2)
r_i.interpolate(reference)


# p = pyvista.Plotter(window_size=[1500, 1000])
p = pyvista.Plotter(window_size=[850, 600], border=False)

# p.subplot(0, 0)  # plot initial guess for displacement
# topology_l, cell_types_l, geometry_l = plot.vtk_mesh(V2)
# grid_l = pyvista.UnstructuredGrid(topology_l, cell_types_l, geometry_l)
# grid_l["U0l"] = U0_i.x.array.reshape((geometry_l.shape[0], 3)) - r_i.x.array.reshape((geometry_l.shape[0], 3))
# # actor_l0 = p.add_mesh(grid_l, style="wireframe", color="w")
# warped_l = grid_l.warp_by_vector("U0l", factor=1.0)
# actor_l1 = p.add_mesh(warped_l, show_edges=False, show_scalar_bar=False)
# p.add_scalar_bar('', vertical=True, position_y=0.5)
# # p.add_text(r"$\mathbf{y}^0_h$", font_size=20, position=(200, 450))
# # p.show_grid()
# p.zoom_camera(1.3)
# # p.show_axes()
# p.save_graphic("Miura_Hessian_deformation.eps")
# p.show()


# p.subplot(1, 0)  # plot initial guess for actuation
# topology_t, cell_types_t, geometry_t = plot.vtk_mesh(VT)
# grid_t = pyvista.UnstructuredGrid(topology_t, cell_types_t, geometry_t)
# grid_t.point_data["t0"] = theta_t.x.array
# warped_t = grid_t.warp_by_scalar("t0", factor=1.0)
# actor_t = p.add_mesh(warped_t, show_edges=False, show_scalar_bar=False, scalars="t0")
# # p.add_text(r'$\theta^0_h$', font_size=20, position=(200, 450))
# p.add_scalar_bar('', vertical=True, position_y=0.52)
# # p.set_scale(xscale=10, yscale=10)
# # p.show_grid()
# p.zoom_camera(1.2)
# # p.show_axes()
# p.save_graphic("Miura_Metric_actuation.eps")
# p.show()
# p.screenshot("Miura_Metric_actuation.png")

topology, cell_types, geometry = plot.vtk_mesh(V2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = uh_i.x.array.reshape((geometry.shape[0], 3)) - r_i.x.array.reshape((geometry.shape[0], 3))
# actor_0 = p.add_mesh(grid, style="wireframe", color="w")
warped = grid.warp_by_vector("u", factor=1.0)
actor_1 = p.add_mesh(warped, show_edges=False, show_scalar_bar=False)
p.add_scalar_bar('', vertical=True, position_y=0.5)
# p.add_text(r"$\mathbf{y}$", font_size=20, position=(250, 400))
# p.show_grid()
p.zoom_camera(1.3)
# p.show_axes()
# p.save_graphic("Miura_converged_deformation.eps")
p.show()

# topology_h, cell_types_h, geometry_h = plot.vtk_mesh(VT)
# grid_h = pyvista.UnstructuredGrid(topology_h, cell_types_h, geometry_h)
# grid_h.point_data["th"] = theta_h.x.array
# warped_h = grid_h.warp_by_scalar("th", factor=1.0)
# actor_h = p.add_mesh(warped_h, show_edges=False, show_scalar_bar=False, scalars="th")
# p.add_scalar_bar('', vertical=True, position_y=0.52)
# # p.add_text(r'$\theta$', font_size=20, position=(200, 450))
# p.zoom_camera(1.2)
# # p.show_axes()
# p.save_graphic("Miura_converged_actuation.eps")
# p.show()

# ## Post-processing
# Get the rotation field
E_1 = as_vector([1.0, 0.0, 0.0])
E_2 = as_vector([0.0, 1.0, 0.0])
E_3 = as_vector([0.0, 0.0, 1.0])
FF = grad(uh)
nn = cross( FF*e_1, FF * e_2 ) / sqrt( inner( cross( FF*e_1 , FF*e_2 ), cross( FF*e_1, FF*e_2 ) ) )
DeG = outer(FF*e_1, E_1) + outer(FF*e_2, E_2) + outer(nn, E_3)  # 3by3 deformation gradient
# DeU = sqrt(DeG.T * DeG)
# R = DeG * inv(DeU)
u_th = sqrt(3.0) * cos((theta_h + phi) / 2)
v_th = 2 * sqrt(2.0 / (5 - 3 * cos(theta_h + phi)))
A_h = as_matrix( [ [ u_th/ u_s, 0.0, 0.0 ], [0.0, v_th/v_s, 0.0], [0.0, 0.0, 1.0] ] )
R = DeG * inv(A_h)
# get skew vector fields
u_0_scaled = a/NN * e_1
v_0_scaled = b/NN * e_2
W_u = inv(R) * dot(grad(R), u_0_scaled)
W_v = inv(R) * dot(grad(R), v_0_scaled)
omega_u = as_vector([W_u[2, 1], W_u[0, 2], W_u[1, 0]])
omega_v = as_vector([W_v[2, 1], W_v[0, 2], W_v[1, 0]])
# get gradient of actuation field
theta_du = dot(grad(theta_h), u_0_scaled)
theta_dv = dot(grad(theta_h), v_0_scaled)

# Interpolate all in DG space
V_T_DG = functionspace(domain, element("DG", domain.basix_cell(),0, shape=(3, 3)))
V_V_DG = functionspace(domain, element("DG", domain.basix_cell(),0, shape=(3,)))
V_S_DG = functionspace(domain, element("DG", domain.basix_cell(),0))

Def_expr = Expression(uh, V_V_DG.element.interpolation_points())
Def_h = Function(V_V_DG)
Def_h.interpolate(Def_expr)

Ang_expr = Expression(theta_h, V_S_DG.element.interpolation_points())
Ang_h = Function(V_S_DG)
Ang_h.interpolate(Ang_expr)

R_expr = Expression(R, V_T_DG.element.interpolation_points())
R_h = Function(V_T_DG)
R_h.interpolate(R_expr)

omega_u_expr = Expression(omega_u, V_V_DG.element.interpolation_points())
omega_u_h = Function(V_V_DG)
omega_u_h.interpolate(omega_u_expr)

omega_v_expr = Expression(omega_v, V_V_DG.element.interpolation_points())
omega_v_h = Function(V_V_DG)
omega_v_h.interpolate(omega_v_expr)

theta_du_expr = Expression(theta_du, V_S_DG.element.interpolation_points())
theta_du_h = Function(V_S_DG)
theta_du_h.interpolate(theta_du_expr)

theta_dv_expr = Expression(theta_dv, V_S_DG.element.interpolation_points())
theta_dv_h = Function(V_S_DG)
theta_dv_h.interpolate(theta_dv_expr)

# ## Export mesh and continuum fields
num_sub_spaces_T = V_T_DG.num_sub_spaces
num_sub_spaces_V = V_V_DG.num_sub_spaces
num_dofs_per_component = int(len(Ang_h.x.array))
vector_Def = np.zeros((num_sub_spaces_V, num_dofs_per_component))
vector_R = np.zeros((num_sub_spaces_T, num_dofs_per_component))
vector_omega_u = np.zeros((num_sub_spaces_V, num_dofs_per_component))
vector_omega_v = np.zeros((num_sub_spaces_V, num_dofs_per_component))
for i in range(num_sub_spaces_V):
    vector_Def[i] = Def_h.sub(i).collapse().x.array
    vector_omega_u[i] = omega_u_h.sub(i).collapse().x.array
    vector_omega_v[i] = omega_v_h.sub(i).collapse().x.array
for i in range(num_sub_spaces_T):
    vector_R[i] = R_h.sub(i).collapse().x.array
xy = V_S_DG.tabulate_dof_coordinates()
vector_Disp = vector_Def.T
vector_omaga_u = vector_omega_u.T
vector_omaga_v = vector_omega_v.T
vector_R = vector_R.T
out = open("FESol_PureBend.csv", "w")
for coord, vec_Def, angle, rotation, omegau, oemgav, thetau, thetav \
        in zip(xy, vector_Disp, Ang_h.x.array, vector_R, vector_omaga_u, vector_omaga_v, theta_du_h.x.array, theta_dv_h.x.array):
    print(f"{coord[0]}, {coord[1]}, {vec_Def[0]}, {vec_Def[1]}, {vec_Def[2]}, {angle}, \
{rotation[0]}, {rotation[1]}, {rotation[2]}, {rotation[3]}, {rotation[4]}, {rotation[5]}, {rotation[6]}, {rotation[7]}, {rotation[8]},\
{omegau[0]}, {omegau[1]}, {omegau[2]}, {oemgav[0]}, {oemgav[1]}, {oemgav[2]}, {thetau}, {thetav}", file=out)
out.close()
