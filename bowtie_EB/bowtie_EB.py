from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_s = sqrt(2.0)
v_s = sqrt(2.0)

# Create mesh
mesh = Mesh('mesh.msh', name='mesh')
N = 40
#mesh = RectangleMesh(N, N, u_s, v_s, diagonal='crossed', name='mesh') #Realistic domain

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Define the boundary conditions
val_theta = 0.1 #max #1.0
x = SpatialCoordinate(mesh)
#Mechanism BC
u_ts_0 = 2 * sin( ( acos( 1-cos( variable(val_theta) ) ) - variable(val_theta) )/2 )
v_ts_0 = 2 * sin( ( acos( 1-cos( variable(val_theta) ) ) + variable(val_theta) )/2 )
bnd = as_vector((u_ts_0 / u_s * x[0], v_ts_0 / v_s * x[1]))
defo = 1.0 - u_ts_0 / u_s #Corresponding imposed deformation
PETSc.Sys.Print('Imposed deformation: %.3e' % defo)

#Interior penalty
alpha = Constant(1e-1) #1e-1 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

# elastic parameters
c_1 = 5 #metric constraint
d_3 = .1
d_2, d_1 = 2e-2, 2e-2

#Nonlinear problem
#Define trial and test functions
test = TestFunction(Z)
w, eta = split(test)

#Define solutions
sol = Function(Z, name='sol')
y, theta = split(sol)

#Interpolate initial guess
sol.sub(0).interpolate(bnd)
sol.sub(1).interpolate(Constant(val_theta))

#Define the boundary conditions
#bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2), DirichletBC(Z.sub(0), bnd, 3), DirichletBC(Z.sub(0), bnd, 4)] #mechanism
bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2)] #bowtie


# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
theta = variable(theta)
u_ts = 2 * sin( ( acos( 1-cos( theta ) ) - theta )/2 )
v_ts = 2 * sin( ( acos( 1-cos( theta ) ) + theta )/2 )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, theta)
v_t_p = diff(v_ts, theta) 

#Preparation for variational form
H = variable(grad(grad(y)))
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)

#Total energy
J = sqrt(det(dot(grad(y).T, grad(y))))
dens = c_1 * inner(L, L)/J + d_3 * theta**2 + d_2 * inner(grad(theta), grad(theta)) + d_1 * inner(H, H)
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
en_pen = inner( dot(avg(G), n('+')), jump(grad(y))) * dS # consistency and symmetry energy term
a -= derivative(en_pen, y, w)
a += alpha / h_avg * inner( jump( grad(y), n ), jump( grad(w), n ) ) * dS #pen term

#Solve
#parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
parameters = {'snes_monitor': None, 'snes_max_it': 25, 'quadrature_degree': '4', 'rtol': 1e-5}
solve(a == 0, sol, bcs=bcs, solver_parameters=parameters)

#plotting the results
aux = Function(V, name='yeff 3d')
aux.interpolate(sol.sub(0)-as_vector((x[0], x[1])))
theta = Function(W, name='theta')
theta.assign(sol.sub(1))
file = VTKFile('res_comp.pvd')
file.write(aux, theta)

##Save results
#with CheckpointFile("res_EB.h5", 'w') as afile:
#    afile.save_mesh(mesh)
#    y = Function(V, name='y')
#    y.assign(sol.sub(0))
#    afile.save_function(y)
#    afile.save_function(theta)
#sys.exit()

#Computing reaction forces
v_reac = Function(Z)
bc_l = DirichletBC(V.sub(0), Constant(1), 1)
bc_l.apply(v_reac.sub(0))
res_l = assemble(action(a, v_reac))
PETSc.Sys.Print('Total force: %.3e' % (res_l / d_3))

#Save forces
comp = 'pinch' #'pinch' #'lr' #'mechanism'

import numpy as np
with open('force_%s.txt' % comp, 'a') as f: #
    np.savetxt(f, np.array([defo, res_l / d_3])[None], delimiter=',', fmt='%.3e')

#Save forces
import numpy as np
with open('actuation_%s.txt' % comp, 'a') as f:
    np.savetxt(f, np.array([defo, min(theta.vector()), max(theta.vector())])[None], delimiter=',', fmt='%.3e')


##Print energies
#J = sqrt(det(dot(grad(y).T, grad(y))))
#en = assemble(c_1 * inner(L, L)/J * dx)
#print(en)
#en = assemble(d_1 * theta ** 2 * dx)
#print(en)
#en = assemble(d_2 * inner(grad(theta), grad(theta)) * dx)
#print(en)
#en = assemble(d_3 * inner(H, H) * dx)
#print(en)
