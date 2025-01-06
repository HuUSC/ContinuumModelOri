from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc
import numpy as np

# Create mesh
mesh = Mesh('mesh.msh', name='mesh')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Define the boundary conditions
val = 0
x = SpatialCoordinate(mesh)
#BC
bnd = as_vector(((1-2*val)*x[0] + val, x[1]))

##Initial guess in displacement
#sol_ig = Function(V, name='IG')
#sol_ig.interpolate(as_vector(((1-2*val)*x[0] + val, x[1])))

#Interior penalty
alpha = Constant(1e-1) #1e-1 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

# elastic parameters
c_1 = 1 #metric constraint
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
#sol.sub(0).interpolate(bnd)

#Load the initial guess from a file
with CheckpointFile("res.h5", 'r') as afile:
    meshR = afile.load_mesh("mesh")
    sol_old = afile.load_function(meshR, "sol", val)
    sol.interpolate(sol_old)
#sys.exit()

#Define the boundary conditions
bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2)]

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
phi = pi/6
u_s = sqrt(3) * cos(phi/2)
v_s = 2 * sqrt( 2/ ( 5-3 * cos(phi) ) )
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
#c_2 = 0 #test
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

##Gradient BC
#a += alpha / h * inner(dot(dot(grad(y), n), n), dot(dot(grad(w), n), n)) * (ds(1) + ds(2)) #lhs pen
#en_pen = inner(dot(dot(G, n), n), dot(grad(y), n)) * (ds(1) + ds(2)) # consistency and symmetry energy term
#a -= derivative(en_pen, y, w)
#a -= alpha / h * inner(dot(dot(grad(bnd), n), n), dot(dot(grad(w), n), n)) * (ds(1) + ds(2))  #rhs pen
#a -= replace(derivative(en_pen, y, w), {y:bnd}) #rhs consistency and symmetry term

#Solve parameters
#parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
parameters = {'snes_monitor': None, 'snes_max_it': 25, 'quadrature_degree': '4', 'rtol': 1e-5}
v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(Z, [v_basis, Z.sub(1)])

disp_max = .45
d_disp = 1e-3
while val < disp_max:
    #Increase disp
    val += d_disp
    PETSc.Sys.Print('Disp: %.3f' % val)

    #Define the boundary conditions
    bnd = as_vector(((1-2*val)*x[0] + val, x[1]))
    bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2)]

    #Solve
    solve(a == 0, sol, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

    #Save results
    with CheckpointFile("res.h5", 'w') as afile:
        afile.save_mesh(mesh)
        afile.save_function(sol, val)

    #Computing reaction forces
    v_reac = Function(Z)
    bc_l = DirichletBC(V.sub(0), Constant(1), 1)
    bc_l.apply(v_reac.sub(0))
    #res_l = assemble(action(a, v_reac))
    #print('Reaction on the left: %.3e' % assemble(res))
    #v_reac.sub(0).interpolate(Constant((0, 0)))
    bc_r = DirichletBC(V.sub(0), Constant(-1), 2)
    bc_r.apply(v_reac.sub(0))
    res_r = assemble(action(a, v_reac))
    res = assemble(action(a, v_reac))
    #print('Reaction on the right: %.3e' % assemble(res))
    #PETSc.Sys.Print('Disp: %.3e' % val)
    #PETSc.Sys.Print('Total force: %.3e' % (abs(res_l)+abs(res_r)))

    #Save forces
    with open('force.txt', 'a') as f:
        np.savetxt(f, np.array([val, res])[None], delimiter=',', fmt='%.3e')
