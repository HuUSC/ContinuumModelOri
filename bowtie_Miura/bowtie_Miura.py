from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
phi = pi/6
u_s = sqrt(3) * cos(phi/2)
v_s = 2 * sqrt( 2/ ( 5-3 * cos(phi) ) )

# Create mesh
#mesh = Mesh('mesh.msh', name='mesh')
N = 40
mesh = RectangleMesh(N, N, u_s, v_s, diagonal='crossed', name='mesh') #Realistic domain

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Define the boundary conditions
val_theta = 2.2 #2.2 max for bowtie #2.5 max for mech lr
x = SpatialCoordinate(mesh)
#Mechanism BC
u_ts_0 = sqrt(3.0) * cos((val_theta + phi) / 2.0)
v_ts_0 = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(val_theta + phi)))
bnd = as_vector((u_ts_0 / u_s * x[0], v_ts_0 / v_s * x[1]))
defo = 1.0 - u_ts_0 / u_s #Corresponding imposed disp
PETSc.Sys.Print('Imposed deformation: %.3e' % defo)

#Interior penalty
alpha = Constant(5e-1) #1e-1 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

# elastic parameters
c_1 = 5 #metric constraint
d_3 = .1 
d_2 = 1e-2 * 1.7**2
d_1 = 1e-2 * 1.7**2

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
bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2), DirichletBC(Z.sub(0), bnd, 3), DirichletBC(Z.sub(0), bnd, 4)] #mechanism
#bcs = [DirichletBC(Z.sub(0), bnd, 1), DirichletBC(Z.sub(0), bnd, 2)] #bowtie


# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_ts = sqrt(3) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2/ ( 5-3 * cos(theta+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
theta = variable(theta) #test
u_t_p = diff(u_ts, theta) #variable(theta) 
v_t_p = diff(v_ts, theta) #variable(theta)

#Preparation for variational form
H = variable(grad(grad(y)))
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)

#Total energy
J = sqrt(det(dot(grad(y).T, grad(y))))
dens = c_1 * inner(L, L)/J + d_3 * theta**2 + d_2 * inner(grad(theta), grad(theta)) + d_1 * inner(H, H) #test
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
a -= inner( dot(avg(G), n('+')), jump(grad(w))) * dS #consistency term
a += alpha / h_avg * inner( jump( grad(y), n ), jump( grad(w), n ) ) * dS #pen term

#Solve
parameters = {'snes_monitor': None, 'snes_max_it': 25, 'quadrature_degree': '4', 'rtol': 1e-5}
solve(a == 0, sol, bcs=bcs, solver_parameters=parameters) #nullspace=nullspace

#plotting the results
aux = Function(V, name='yeff 3d')
aux.interpolate(sol.sub(0)-as_vector((x[0], x[1])))
theta = Function(W, name='theta')
theta.assign(sol.sub(1))
file = VTKFile('res_comp.pvd')
file.write(aux, theta)

##Test
#with CheckpointFile("res_test.h5", 'w') as afile:
#    afile.save_mesh(mesh)
#    afile.save_function(sol)
#sys.exit()

##Save results
#with CheckpointFile("res_Miura.h5", 'w') as afile:
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

#Print energies
J = sqrt(det(dot(grad(y).T, grad(y))))
en = assemble(c_1 * inner(L, L)/J * dx)
print('W_1 = %.2e' % en)
en = assemble(d_3 * theta ** 2 * dx)
print('theta = %.2e' % en)
en = assemble(d_2 * inner(grad(theta), grad(theta)) * dx)
print('grad theta = %.2e' % en)
en = assemble(d_1 * inner(H, H) * dx)
print('Hessian = %.2e' % en)
en = assemble(.5 * alpha / h_avg * inner( jump( grad(y), n ), jump( grad(y), n ) ) * dS)
print('Pen = %.2e' % en)

#Save forces
comp = 'mech' #'lr' #'mech' #'bowtie'
import numpy as np
with open('force_%s.txt' % comp, 'a') as f:
    np.savetxt(f, np.array([defo, res_l / d_3])[None], delimiter=',', fmt='%.3e')

#Save actuation
import numpy as np
with open('actuation_%s.txt' % comp, 'a') as f:
    np.savetxt(f, np.array([defo, min(theta.vector()), max(theta.vector())])[None], delimiter=',', fmt='%.3e')
