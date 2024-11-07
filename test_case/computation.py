from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc

# Create mesh
#N = 30
#mesh = UnitSquareMesh(N, N, diagonal='crossed')
mesh = Mesh('mesh.msh')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

# Define boundary condition
x = SpatialCoordinate(mesh)
val = .2
u1 = as_vector((x[0] + val, x[1], 0))
u2 = as_vector((x[0] - val, x[1], 0))
bcs = [DirichletBC(V, u1, 1), DirichletBC(V, u2, 2)]

#Interior penalty
alpha = Constant(1e1) #1e2 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

#Bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(grad(u)), grad(grad(v)))*dx \
  - inner(dot(avg(grad(grad(u))), n('+')), jump(grad(v)))*dS \
  - inner(jump(grad(u)), dot(avg(grad(grad(v))), n('+')))*dS \
  + alpha/h_avg*inner(jump(grad(u)), jump(grad(v)))*dS

#Penalty term for the gradient Dirichlet bc
a += alpha/h * inner(dot(grad(u), n), dot(grad(v), n)) * (ds(1) + ds(2))

#Rhs boundary penalty term
G1 = Constant((-1, 0, -1)) #/10 # / sqrt(2)
G2 = Constant((1, 0, -1)) #/10 #/ sqrt(2)
L = alpha/h * inner(G1, dot(grad(v), n)) * ds(1) - inner(G1, dot(dot(grad(grad(v)), n), n)) * ds(1)
L += alpha/h * inner(G2, dot(grad(v), n)) * ds(2) - inner(G2, dot(dot(grad(grad(v)), n), n)) * ds(2)

#Lhs boundary penalty term
a -= inner(dot(grad(u), n), dot(dot(grad(grad(v)), n), n)) * ds(2) + inner(dot(grad(v), n), dot(dot(grad(grad(u)), n), n)) * ds(2)
a -= inner(dot(grad(u), n), dot(dot(grad(grad(v)), n), n)) * ds(1) + inner(dot(grad(v), n), dot(dot(grad(grad(u)), n), n)) * ds(1)

# Solve variational problem
sol_ig = Function(V, name='IG')
#v_basis = VectorSpaceBasis(constant=True)
solve(a == L, sol_ig, bcs, solver_parameters={'quadrature_degree': '2'})

# Save solution to file
file = VTKFile("IG.pvd")
WW = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
aux = Function(WW, name='IG')
x = SpatialCoordinate(mesh)
aux.interpolate(sol_ig - as_vector((x[0], x[1], 0)))
file.write(aux)
sys.exit()

#Compute initial guess for the angle field
theta_ig = Function(W, name='IG theta')
# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
e_1 = Constant((1, 0))
e_2 = Constant((0, 1))
phi = pi/6
u_s = sqrt(3.0) * cos(phi/2)
v_s = 2 * sqrt( 2.0/ ( 5-3 * cos(phi) ) )
u_0 = u_s * e_1
v_0 = v_s * e_2
u_ts = sqrt(3.0) * cos( (theta_ig+phi)/2 )
v_ts = 2 * sqrt( 2.0/ ( 5-3 * cos(theta_ig+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )

#defining the energy to minimize
L = dot(grad(sol_ig).T, grad(sol_ig)) - dot(A_t.T, A_t)
c = 1 #coercivity constant
energy = inner(L, L) * dx + c * inner(grad(theta_ig), grad(theta_ig)) * dx
zeta = TestFunction(W)
a = derivative(energy, theta_ig, zeta)

#Solve
solve(a == 0, theta_ig, solver_parameters={'quadrature_degree': '2'}) #, solver_parameters={'snes_monitor': None, 'snes_max_it': 10}) #bcs=bcs

#Output IG in theta
file = VTKFile("IG_theta.pvd")
file.write(theta_ig)
#sys.exit()

PETSc.Sys.Print('Initial guess ok!\n')


#Nonlinear problem
#Define trial and test functions
test = TestFunction(Z)
w, eta = split(test)

#Define solutions
sol = Function(Z, name='sol')
y, theta = split(sol)

#Interpolate initial guess
sol.sub(0).interpolate(sol_ig)
#Go get code from Hu for the computation of theta
sol.sub(1).interpolate(theta_ig)

#Define the boundary conditions
bcs = [DirichletBC(Z.sub(0), u1, 1), DirichletBC(Z.sub(0), u2, 2)]

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_ts = sqrt(3.0) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2.0/ ( 5-3 * cos(theta+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, sol) #variable(theta))
v_t_p = diff(v_ts, sol) #variable(theta))

#Preparation for variational form
H = variable( grad(grad(y)) )
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)
q = v_t_p * v_ts * inner( H, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( H,outer(N,v_0,v_0) )

# elastic parameters
#c_1, c_2, d_1, d_2, d_3 = 1, .5, .1, 1e-2, 1e-2
c_1, c_2, d_1, d_2, d_3 = 1, 1, 1e-2, 1e-1, 1e-2


#Total energy
dens = c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( H, H)
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
en_pen = inner( dot(avg(G), n('+')), jump(grad(y))) * dS # consistency and symmetry energy term
a -= derivative(en_pen, y, w)
a += alpha / h_avg * inner(jump(grad(y)), jump(grad(w))) * dS #pen term

##Gradient Dirichlet bc - lhs
#a += alpha/h * inner(dot(grad(y), n), dot(grad(w), n)) * (ds(1) + ds(2)) #LS pen term
#a -= inner(dot(grad(y), n), dot(dot(grad(grad(w)), n), n)) * (ds(1) + ds(2)) + inner(dot(grad(w), n), dot(dot(grad(grad(y)), n), n)) * (ds(1) + ds(2)) #consistency and sym term
#
##Gradient Dirichlet BC - rhs
#a -= alpha / h * inner(G1, dot(grad(w), n) ) * ds(1) + alpha / h * inner(G2, dot(grad(w), n) ) * ds(2) #LS pen term
#a += inner(G1, dot(dot(grad(grad(w)), n), n)) * ds(1) + inner(G2, dot(dot(grad(grad(w)), n), n)) * ds(2) #sym term

#Solve
parameters = {'snes_monitor': None, 'snes_max_it': 10, 'quadrature_degree': '4', 'rtol': 1e-5, 'ksp_type': 'preonly', 'pc_type': 'lu'}
solve(a == 0, sol, bcs=bcs, solver_parameters=parameters)


#plotting the results
aux = Function(WW, name='yeff 3d')
aux.interpolate(sol.sub(0)-as_vector((x[0], x[1], 0)))
file = VTKFile('surf_comp.pvd')
file.write(aux)

aux = Function(W, name='theta')
aux.assign(sol.sub(1))
file = VTKFile('theta_comp.pvd')
file.write(aux)
