from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc
import numpy as np

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
phi = pi/6
u_s = sqrt(3) * cos(phi/2)
v_s = 2 * sqrt( 2/ ( 5-3 * cos(phi) ) )

# Create mesh
#mesh = Mesh('mesh.msh', name='mesh')
N = 30
l, w = u_s, v_s
#mesh = RectangleMesh(N, N, l, w, diagonal='crossed', name='mesh')
mesh = UnitSquareMesh(N, N, diagonal='crossed', name='mesh')

##Load mesh
#with CheckpointFile("res.h5", 'r') as afile:
#    mesh = afile.load_mesh("mesh")

#Save mesh to file
with CheckpointFile("res_mech.h5", 'w') as afile:
    afile.save_mesh(mesh)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Interior penalty
alpha = Constant(1e-1) #1e-1 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

# elastic parameters
c_1 = 5 #metric constraint
d_1 = .1
d_2, d_3 = 1e-2, 1e-2

#Nonlinear problem
#Define trial and test functions
test = TestFunction(Z)
w, eta = split(test)

#Define solutions
sol = Function(Z, name='sol')
y, theta = split(sol)
        
#Interpolate initial guess
x = SpatialCoordinate(mesh)
#val =.1
#bnd = as_vector(((1-2*val)*x[0] + val, x[1]))
sol.sub(0).interpolate(x)

##Load the initial guess from a file
#idx = 500
#with CheckpointFile("res_mech.h5", 'r') as afile:
#    sol_old = afile.load_function(mesh, "sol", idx=idx)
#    sol.sub(0).interpolate(as_vector((sol_old.sub(0)[0], sol_old.sub(0)[1])))
#    sol.sub(1).assign(sol_old.sub(1))
##sys.exit()

##test output
#file = VTKFile('test.pvd')
#file.write(sol.sub(0), sol.sub(1))
#sys.exit()

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_ts = sqrt(3) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2/ ( 5-3 * cos(theta+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, sol) 
v_t_p = diff(v_ts, sol) 

#Preparation for variational form
H = variable(grad(grad(y)))
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)
#q = v_t_p * v_ts * inner( H, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( H,outer(N,v_0,v_0) )

#Total energy
J = sqrt(det(dot(grad(y).T, grad(y))))
#dens = c_1 * inner(L, L) + d_1 * theta**2 + d_2 * inner(grad(theta), grad(theta)) + d_3 * inner(H, H)
dens = c_1 * inner(L, L)/J + d_1 * theta**2 + d_2 * inner(grad(theta), grad(theta)) + d_3 * inner(H, H) #test
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
en_pen = inner( dot(avg(G), n('+')), jump(grad(y))) * dS # consistency and symmetry energy term
a -= derivative(en_pen, y, w)
a += alpha / h_avg * inner( jump( grad(y), n ), jump( grad(w), n ) ) * dS #pen term


#Solve parameters
#parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", 'quadrature_degree': '4', 'rtol': 1e-5}
parameters = {'snes_monitor': None, 'snes_max_it': 25, 'quadrature_degree': '4', 'rtol': 1e-10}
v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(Z, [v_basis, Z.sub(1)])

disp_max = .42
d_disp = 1e-2
idx = -1
val = idx * d_disp
while val < disp_max:
    #Increase disp
    val += d_disp
    idx += 1
    PETSc.Sys.Print('Disp: %.4f' % val)

    #Define the boundary conditions for the mechanism motion
    u_ts_0 = sqrt(3.0) * cos((val + phi) / 2.0)
    v_ts_0 = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(val + phi)))
    bnd = as_vector((u_ts_0 / u_s * x[0], v_ts_0 / v_s * x[1]))
    bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2), DirichletBC(Z.sub(0), bnd, 3), DirichletBC(Z.sub(0), bnd, 4)]
    #bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2)]

    #Solve
    solve(a == 0, sol, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

    #Save results
    with CheckpointFile("res_mech.h5", 'a') as afile:
        afile.save_function(sol, idx=idx)

    #Computing reaction forces
    v_reac = Function(Z)
    bc_l = DirichletBC(V.sub(0), Constant(1), 1)
    bc_l.apply(v_reac.sub(0))
    res_l = assemble(action(a, v_reac))

    #Save forces
    with open('force_mech.txt', 'a') as f:
        np.savetxt(f, np.array([2*val, res_l])[None], delimiter=',', fmt='%.3e')
