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
import vtk
vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()
# vtk_mathtext.MathTextIsSupported()

# domain, mesh
phi = pi / 6.0
u_s = sqrt(3.0) * cos(phi / 2.0)
v_s = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(phi)))
a, b = u_s/v_s , 1.0
# a, b = 1.0 , 1.0
# print(a)
NN = 30
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=( (0.0, 0.0), (a, b) ), n=(NN, NN),
                            cell_type=CellType.quadrilateral,
                            ghost_mode=GhostMode.shared_facet)

# Vu = element("P", domain.basix_cell(),2, shape=(3,))
# Vt = element("P", domain.basix_cell(),1)
# MX = functionspace(domain, mixed_element([Vu, Vt]))
#
# # get subspace of MX
# V0 = MX.sub(0)
# Q, Q_map = V0.collapse()
# # Q0 = MX.sub(0).sub(1) # subspace for x-component of displacement
# # Qx, _ = Q0.collapse()
# # Q2 = MX.sub(0).sub(2) # subspace for z-component of displacement
# # Qz, _ = Q2.collapse()
# V1 = MX.sub(1)
# R, R_map = V1.collapse()

# boundary conditions of displacement
def boundary_zero(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CL(x):
    return np.vstack( ( np.ones_like(x[0]) * (0.15), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_CR(x):
    return np.vstack( ( np.ones_like(x[0]) * (-0.15), np.zeros_like(x[0]), np.zeros_like(x[0]) ))
# def boundary_CR(x):
#     return np.vstack( ( np.ones_like(x[0]) * (-0.2), np.zeros_like(x[0]), (x[1]-b/2)**2 ))

def boundary_center(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]) * (0.05) ))

# gradient Dirichlet boundary conditions
def GR_b(x):
    return np.vstack( ( np.ones_like(x[0])*(1.0/sqrt(2)), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0/sqrt(2)) ) )

# def GR_b(x):
#     return np.vstack( ( np.ones_like(x[0])*(1.0), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0) ) )

def GL_b(x):
    return np.vstack( ( np.ones_like(x[0])*(-1.0/sqrt(2)), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0/sqrt(2)) ) )

# def GL_b(x):
#     return np.vstack( ( np.ones_like(x[0])*(-1.0), np.zeros_like(x[0]), np.ones_like(x[0])*(-1.0) ) )


# ## Form displacement Dirichlet boundary conditions
fdim = domain.topology.dim - 1

# facets_CR = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], a), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
facets_CR = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], a) )
# dofs_CR = locate_dofs_topological((V0, Q), 1, facets_CR)
# u_CR = Function(Q)
# u_CR.interpolate(boundary_CR)
# bc_CR = dirichletbc(u_CR, dofs_CR, V0)

# facets_CL = mesh.locate_entities_boundary( domain, 1, lambda x: np.logical_and( np.isclose( x[0], 0.0), np.logical_and( x[1] <= b/2 + b/NN, b/2 - b/NN <= x[1] ) ) )
facets_CL = mesh.locate_entities_boundary( domain, 1, lambda x: np.isclose( x[0], 0.0) )
# facets_CL = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], 0.0), np.isclose( x[1], b/2) ) )
# dofs_CL = locate_dofs_topological((V0, Q), 1, facets_CL)
# u_CL = Function(Q)
# u_CL.interpolate(boundary_CL)
# bc_CL = dirichletbc(u_CL, dofs_CL, V0)

# facets_whole = mesh.locate_entities_boundary(domain, fdim, lambda x: np.logical_or.reduce((np.isclose(x[0],0.0),
#                     np.isclose(x[0], a), np.isclose(x[1], 0.0), np.isclose(x[1], b))))
# dofs_whole = locate_dofs_topological((V0, Q), fdim, facets_whole)
# u_whole= Function(Q)
# u_whole.interpolate(boundary_zero)
# bc_whole = dirichletbc(u_whole, dofs_whole, V0)
#
facets_C = mesh.locate_entities( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1], b/2)) )
# dofs_C = locate_dofs_topological((V0, Q), 0, facets_C)
# u_C = Function(Q)
# u_C.interpolate(boundary_center)
# bc_C = dirichletbc(u_C, dofs_C, V0)

# Collect displacement Dirichlet boundary conditions
# bcs = [bc_CR]
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
h = CellDiameter(domain)  # cell diameter
n_F = FacetNormal(domain)  # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet

# ####### initial guess for displacement field from 3D bi-harmonic
VL = functionspace(domain, element("P", domain.basix_cell(),2, shape=(3,)))
# VL = Q
ul = TrialFunction(VL)
wl = TestFunction(VL)
ul_h = TrialFunction(VL)
wl_h = TestFunction(VL)

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

alpha_l, alpha_d = 3.0, 2  # interior, boundary
# increasing alpha_d makes self-intersect

IPTL = alpha_l / h_avg * inner( jump( grad(ul), n_F ), jump( grad(wl), n_F ) ) \
       - inner( avg( div(grad(ul)) ) , jump( grad(wl), n_F) ) - inner( avg( div(grad(wl)) ) , jump( grad(ul), n_F) )

GR_be_l = Function(VL)
GR_be_l.interpolate(GR_b)
GL_be_l = Function(VL)
GL_be_l.interpolate(GL_b)

GDBC_RL_bi = alpha_d / h * inner( dot( grad(ul), n_F), dot( grad(wl), n_F) ) \
             - inner( dot( grad(wl), n_F),  div(grad(ul)) ) - inner( dot( grad(ul), n_F),  div(grad(wl)) )

GDBC_RL_li = alpha_d / h * inner( dot( I, n_F) - GR_be_l, dot( grad(wl), n_F) ) \
             - inner( dot( I, n_F) - GR_be_l, div(grad(wl)) )

GDBC_LL_bi = alpha_d / h * inner(dot(grad(ul), n_F), dot(grad(wl), n_F)) \
             - inner(dot(grad(wl), n_F), div(grad(ul))) - inner(dot(grad(ul), n_F), div(grad(wl)))

GDBC_LL_li = alpha_d / h * inner(dot(I, n_F) - GL_be_l, dot(grad(wl), n_F)) \
             - inner(dot(I, n_F) - GL_be_l, div(grad(wl)))

f = fem.Constant(domain, default_scalar_type((0, 0, 0)))

a = 2 * inner( div(grad(ul)), div(grad(wl)) ) * dx + IPTL * dS + GDBC_RL_bi * ds(2) + GDBC_LL_bi * ds(1)
L = - GDBC_RL_li * ds(2) - GDBC_LL_li * ds(1)

# a = 2 * inner( div(grad(ul)), div(grad(wl)) ) * dx + IPTL * dS
# L = inner(f, wl) * dx

problem = LinearProblem(a, L, bcs=bcs_l, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
U0 = problem.solve()

# ## plot the deformation

# ####### initial guess for displacement field from 3D bi-Hessian
IPTL_h = alpha_l / h_avg * inner( jump( grad(ul_h), n_F ), jump( grad(wl_h), n_F ) ) \
       - inner( avg( dot( dot( grad(grad(ul_h)), n_F), n_F) ) , jump( grad(wl_h), n_F) )\
       - inner( avg( dot( dot( grad(grad(wl_h)), n_F), n_F) ) , jump( grad(ul_h), n_F) )

GDBC_RL_bi_h = alpha_d / h * inner( dot( grad(ul_h), n_F), dot( grad(wl_h), n_F) ) \
             - inner( dot( grad(wl_h), n_F), dot( dot( grad(grad(ul_h)), n_F), n_F) )\
             - inner( dot( grad(ul_h), n_F), dot( dot( grad(grad(wl_h)), n_F), n_F) )

GDBC_RL_li_h = alpha_d / h * inner( dot( I, n_F) - GR_be_l, dot( grad(wl_h), n_F) ) \
             - inner( dot( I, n_F) - GR_be_l, dot( dot( grad(grad(wl_h)), n_F), n_F) )

GDBC_LL_bi_h = alpha_d / h * inner( dot( grad(ul_h), n_F), dot( grad(wl_h), n_F) ) \
             - inner( dot( grad(wl_h), n_F), dot( dot( grad(grad(ul_h)), n_F), n_F) )\
             - inner( dot( grad(ul_h), n_F), dot( dot( grad(grad(wl_h)), n_F), n_F)  )

GDBC_LL_li_h = alpha_d / h * inner(dot(I, n_F) - GL_be_l, dot(grad(wl_h), n_F)) \
             - inner(dot(I, n_F) - GL_be_l, dot( dot( grad(grad(wl_h)), n_F), n_F) )

a_h = 2 * inner( grad(grad(ul_h)), grad(grad(wl_h)) ) * dx + IPTL_h * dS + GDBC_RL_bi_h * ds(2) + GDBC_LL_bi_h * ds(1)
L_h = - GDBC_RL_li_h * ds(2) - GDBC_LL_li_h * ds(1)

# a_h = 2 * inner( grad(grad(ul_h)), grad(grad(wl_h)) ) * dx + IPTL_h * dS
# L_h = inner(f, wl_h) * dx

problem_h = LinearProblem(a_h, L_h, bcs=bcs_l, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
U0_h = problem_h.solve()

V2 = functionspace(domain, element("P", domain.basix_cell(),1, shape=(3,)))
U0_i = Function(V2)
U0_i.interpolate(U0)
U0_h_i = Function(V2)
U0_h_i.interpolate(U0_h)

p = pyvista.Plotter(shape=(1, 2), window_size=[2000, 1000])


p.subplot(0, 0)  # plot initial guess for displacement
topology_l, cell_types_l, geometry_l = plot.vtk_mesh(V2)
grid_l = pyvista.UnstructuredGrid(topology_l, cell_types_l, geometry_l)
grid_l["U0l"] = U0_i.x.array.reshape((geometry_l.shape[0], 3))
actor_l0 = p.add_mesh(grid_l, style="wireframe", color="k")
warped_l = grid_l.warp_by_vector("U0l", factor=1.5)
actor_l1 = p.add_mesh(warped_l, show_edges=False, show_scalar_bar=True)
p.add_text('Bi-Harmonic')
p.show_grid(bounds=[0, 1, 0, 1, 0, 1], n_xlabels=6, n_ylabels=6, n_zlabels=6)
p.show_axes()
# p.show_bounds(bounds=[0, 1, 0, 1, 0, 1])

p.subplot(0, 1)  # plot displacement field
topology, cell_types, geometry = plot.vtk_mesh(V2)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
grid["u"] = U0_h_i.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=False, show_scalar_bar=True)
p.add_text('Bi-Hessian')
p.show_grid(bounds=[0, 1, 0, 1, 0, 1], n_xlabels=6, n_ylabels=6, n_zlabels=6)
p.show_axes()
# p.show_bounds(bounds=[0, 1, 0, 1, 0, 1], axes_ranges=[0, 1, 0, 1, 0, 1])

p.show()
p.screenshot("Biharmonic_BiHessian.png")