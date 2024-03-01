from dolfinx import log, default_scalar_type, mesh, io, plot, default_real_type, fem
from dolfinx.fem.petsc import NonlinearProblem
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
a, b = sqrt(3), 2.0
domain = mesh.create_rectangle(comm=MPI.COMM_WORLD, points=( (0.0, 0.0), (a, b) ), n=(30, 30),
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


#get subspace of MX
V0 = MX.sub(0)
Q, _ = V0.collapse()
# Q0  = MX.sub(0).sub(0)
# Qu, _ = Q0.collapse()
# V1 = MX.sub(1)
# R, _ = V1.collapse()

# boundary conditions of displacement
def boundary_zero(x):
    return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))

def boundary_right(x):
    return np.vstack( ( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])*(0.3) ))

def boundary_left(x):
    return np.vstack(( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])*(0.3) ))

def boundary_top(x):
    return np.vstack(( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

def boundary_bottom(x):
    return np.vstack(( np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]) ))

fdim = domain.topology.dim - 1

# facets_whole = mesh.locate_entities_boundary( domain, fdim, lambda x: np.logical_or.reduce((np.isclose( x[0],0.0),
#                         np.isclose( x[0],1.0), np.isclose( x[1], 0.0), np.isclose( x[1], 1.0) )) )
# dofs_whole = locate_dofs_topological((V0, Q), fdim, facets_whole)
# u_whole= Function(Q)
# u_whole.interpolate(boundary_zero)
# bc_whole = dirichletbc(u_whole, dofs_whole, V0)

facets_R = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a), np.isclose( x[1], b)) )
dofs_R = locate_dofs_topological((V0, Q), 0, facets_R)
u_R= Function(Q)
u_R.interpolate(boundary_right)
bc_R = dirichletbc(u_R, dofs_R, V0)

facets_L = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0],0.0), np.isclose( x[1],0.0)) )
dofs_L = locate_dofs_topological((V0, Q), 0, facets_L)
u_L = Function(Q)
u_L.interpolate(boundary_left)
bc_L = dirichletbc(u_L, dofs_L, V0)

facets_T = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1], b)) )
dofs_T = locate_dofs_topological((V0, Q), 0, facets_T)
u_T= Function(Q)
u_T.interpolate(boundary_top)
bc_T = dirichletbc(u_T, dofs_T, V0)

facets_B = mesh.locate_entities_boundary( domain, 0, lambda x: np.logical_and( np.isclose( x[0], a/2), np.isclose( x[1],0.0)) )
dofs_B = locate_dofs_topological((V0, Q), 0, facets_B)
u_B = Function(Q)
u_B.interpolate(boundary_bottom)
bc_B = dirichletbc(u_B, dofs_B, V0)

# Collect Dirichlet boundary conditions
bcs=[bc_T, bc_B, bc_R, bc_L]
# bcs=[bc_T, bc_B]

# elasticity parameters
c_1, c_2, d_1, d_2, d_3 = 1.0, 1.0, 1E-2, 1E-2, 1E-2

# theta = variable(theta)

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
e_1 = as_vector([1.0, 0.0])
e_2 = as_vector([0.0, 1.0])
phi = pi/6
u_s = sqrt(3.0) * cos(phi/2)
v_s = 2 * sqrt( 2.0/ ( 5-3 * cos(phi) ) )
u_0 = u_s * e_1
v_0 = v_s * e_2
u_ts = sqrt(3.0) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2.0/ ( 5-3 * cos(theta+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, variable(theta))
v_t_p = diff(v_ts, variable(theta))
I = as_matrix( [ [1.0, 0.0], [0.0, 1.0], [0.0, 0.0] ] )


# variational form
F =  I + grad(u)  # variable grad(u)
G = variable( grad( grad(u) ) )  # variable grad(grad(u))
n = cross( F*e_1, F * e_2 ) / sqrt( inner( cross( F*e_1 , F*e_2 ), cross( F*e_1, F*e_2 ) ) )
# n = cross( F[:,0], F[:,1] ) / sqrt( inner( cross( F[:,0], F[:,1] ), cross( F[:,0], F[:,1] )  ) )
L = F.T * F - A_t.T * A_t
q = v_t_p * v_ts * inner( G, outer(n,u_0,u_0)  ) \
    + u_t_p * u_ts * inner( G,outer(n,v_0,v_0) )
# H =  2 * c_2 * q * ( inner( v_t_p , v_t) * outer(n,u_0,u_0) + inner( u_t_p , u_t) * outer(n,v_0,v_0) ) \
#      + 2 * d_3 * G
# print(F)

# energy density
W = c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( G, G )
H = diff( W, G )
# Total potential energy
Ee = W * dx
# first variation of Ee
Ee_Var = derivative(Ee, U, V)

# interior penalty terms
alpha = default_scalar_type(0.1) # penalty parameter
h = CellDiameter(domain) # cell diameter
n_F = FacetNormal(domain) # outward-facing normal vector
h_avg = avg(h)  # average size of cells sharing a facet
IPT =  alpha / h_avg * inner( jump( grad(u), n_F ), jump( grad(w), n_F ) ) \
        - inner( avg( dot( dot( H, n_F), n_F) ) , jump( grad(w), n_F) ) - \
     inner( avg( dot( grad(w), n_F ) ) , jump( dot( H, n_F), n_F) )

# weak form
WF = Ee_Var + IPT * dS
# WF = Ee_Var
J = derivative(WF, U)
residual = form(WF)
jacobian = form(J)

dU = Function(MX)
A = fem.petsc.create_matrix(jacobian)
L = fem.petsc.create_vector(residual)
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)

max_iterations = 100

i = 0
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

    if correction_norm < 1e-6:
        break

# # solve the problem
# problem = NonlinearProblem( WF, U, bcs)
#
# solver = NewtonSolver(MPI.COMM_WORLD, problem)
#
# # Set Newton solver options
# # solver.atol = 1E-06
# # solver.rtol = 1E-06
#
# solver.convergence_criterion = "incremental"
# solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
#
# # We can customize the linear solver used inside the NewtonSolver by
# # modifying the PETSc options
# ksp = solver.krylov_solver
# opts = PETSc.Options()  # type: ignore
# option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "preonly"
# opts[f"{option_prefix}pc_type"] = "lu"
# sys = PETSc.Sys()  # type: ignore
# # For factorisation prefer MUMPS, then superlu_dist, then default.
# if sys.hasExternalPackage("mumps"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# elif sys.hasExternalPackage("superlu_dist"):
#     opts[f"{option_prefix}pc_factor_mat_solver_type"] = "umfpack"
# ksp.setFromOptions()


# uh, theta_h = U.sub(0).collapse(), U.sub(1).collapse()
uh, theta_h = U.sub(0).collapse(), U.sub(1).collapse()
print(uh.x.array)

#

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
else:
    figure_as_array = p.screenshot("deflection.png")