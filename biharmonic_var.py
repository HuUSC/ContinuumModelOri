import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, GhostMode
from ufl import (CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner,
                 jump, pi, sin)
from dolfinx.fem import dirichletbc, locate_dofs_topological, functionspace, Function, locate_dofs_geometrical

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from basix.ufl import element
from ufl import *
import pyvista
# -

# We begin by using {py:func}`create_rectangle
# <dolfinx.mesh.create_rectangle>` to create a rectangular
# {py:class}`Mesh <dolfinx.mesh.Mesh>` of the domain, and creating a
# finite element {py:class}`FunctionSpaceBase <dolfinx.fem.FunctionSpaceBase>`
# $V$ on the mesh.

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(32, 32),
                            cell_type=CellType.triangle,
                            ghost_mode=GhostMode.shared_facet)

Ve = element("P", msh.basix_cell(),2, shape=(3,))
V = functionspace(msh, Ve)
x = ufl.SpatialCoordinate(msh)

# Next, we locate the mesh facets that lie on the boundary
# $\Gamma_D = \partial\Omega$.
# We do this using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>` and providing  a marker
# function that returns `True` for points `x` on the boundary and
# `False` otherwise.
def boundary_zero(x):
    return np.vstack((np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])))
def boundary_right(x):
    return np.vstack(( np.ones_like(x[0])*0.2, np.zeros_like(x[0]), np.zeros_like(x[0])))

def boundary_center(x):
    return np.vstack(( np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])*0.2))

dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
u_L= Function(V)
u_L.interpolate(boundary_zero)
bc_L = dirichletbc(u_L, dofs_L)

dofs_R = locate_dofs_geometrical(V, lambda x: np.logical_and(np.isclose( x[0],1.0), np.isclose( x[1], 0.5) ))
u_R= Function(V)
u_R.interpolate(boundary_right)
bc_R = dirichletbc(u_R, dofs_R)

dofs_whole = locate_dofs_geometrical(V, lambda x: np.logical_or.reduce((np.isclose( x[0],0.0), np.isclose( x[0],1.0), np.isclose( x[1], 0.0), np.isclose( x[1], 1.0) )))
u_whole= Function(V)
u_whole.interpolate(boundary_zero)
bc_whole = dirichletbc(u_whole, dofs_whole)

dofs_C = locate_dofs_geometrical(V, lambda x: np.logical_and(np.isclose( x[0],0.5), np.isclose( x[1], 0.5) ))
u_C= Function(V)
u_C.interpolate(boundary_center)
bc_C = dirichletbc(u_C, dofs_C)


# Next, we express the variational problem using UFL.
#
# First, the penalty parameter $\alpha$ is defined. In addition, we define a
# variable `h` for the cell diameter $h_E$, a variable `n`for the
# outward-facing normal vector $n$ and a variable `h_avg` for the
# average size of cells sharing a facet
# $\left< h \right> = \frac{1}{2} (h_{+} + h_{-})$. Here, the UFL syntax
# `('+')` and `('-')` restricts a function to the `('+')` and `('-')`
# sides of a facet.

alpha = ScalarType(4.0)
h = CellDiameter(msh)
n = FacetNormal(msh)
h_avg = avg(h)

# After that, we can define the variational problem consisting of the bilinear
# form $a$ and the linear form $L$. The source term is prescribed as
# $f = 4.0 \pi^4\sin(\pi x)\sin(\pi y)$. Note that with `dS`, integration is
# carried out over all the interior facets $\mathcal{E}_h^{\rm int}$, whereas
# with `ds` it would be only the facets on the boundary of the domain, i.e.
# $\partial\Omega$. The jump operator
# $[\!\![ w ]\!\!] = w_{+} \cdot n_{+} + w_{-} \cdot n_{-}$ w.r.t. the
# outward-facing normal vector $n$ is in UFL available as `jump(w, n)`.

# +
# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, default_scalar_type((0, 0, 0)))

a = inner(grad(grad(u)), grad(grad(v))) * dx \
    - inner( avg( dot( dot( grad(grad(u)) , n), n) ), jump(grad(v), n)) * dS \
    - inner( avg( dot( dot( grad(grad(v)) , n), n) ), jump(grad(u), n)) * dS \
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
L = inner(f, v) * dx

# We create a {py:class}`LinearProblem <dolfinx.fem.petsc.LinearProblem>`
# object that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case we use a direct (LU) solver. The {py:func}`solve
# <dolfinx.fem.petsc.LinearProblem.solve>` will compute a solution.

problem = LinearProblem(a, L, bcs=[bc_C, bc_whole], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# The solution can be written to a  {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with ParaView or VisIt

# with io.XDMFFile(msh.comm, "out_biharmonic/biharmonic.xdmf", "w") as file:
#     V1 = fem.functionspace(msh, ("Lagrange", 1))
#     u1 = fem.Function(V1)
#     u1.interpolate(uh)
#     file.write_mesh(msh)
#     file.write_function(u1)

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
# try:
#     import pyvista
#     cells, types, x = plot.vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(cells, types, x)
#     grid.point_data["u"] = uh.x.array.real
#     grid.set_active_scalars("u")
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     if pyvista.OFF_SCREEN:
#         pyvista.start_xvfb(wait=0.1)
#         plotter.screenshot("uh_biharmonic_var.png")
#     else:
#         plotter.show()
# except ModuleNotFoundError:
#     print("'pyvista' is required to visualise the solution")
#     print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
#+

pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")