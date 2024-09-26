from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc

#parameters.parameters = {'quadrature_degree': '2'}

# initial conditions for \theta
phi = pi/2

# Create mesh
#N = 20 #80 computation #10 #debug
#mesh = UnitSquareMesh(N, N, diagonal='crossed')
mesh = Mesh('mesh.msh')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Define the boundary conditions
val = .3
x = SpatialCoordinate(mesh)
#BC
boundary_CL = as_vector((x[0] + val, x[1], 0))
boundary_CR = as_vector((x[0] - val, x[1], 0))
bcs = [DirichletBC(V, boundary_CL, 1), DirichletBC(V, boundary_CR, 2)]

#Interior penalty
alpha = Constant(1e2) #1e2 #10 #penalty parameter
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

#Initial guess
#Bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(grad(u)), grad(grad(v)))*dx \
  - inner(dot(avg(grad(grad(u))), n('+')), jump(grad(v)))*dS \
  - inner(jump(grad(u)), dot(avg(grad(grad(v))), n('+')))*dS \
  + alpha/h_avg*inner(jump(grad(u)), jump(grad(v)))*dS

#Linear form
L = Constant(0) * v[0] * dx

# Solve variational problem
sol_ig = Function(V, name='IG')
nullspace = VectorSpaceBasis(constant=True)
solve(a == L, sol_ig, bcs, nullspace=nullspace)

# Save solution to file
file = VTKFile("IG.pvd")
aux = Function(V, name='IG')
x = SpatialCoordinate(mesh)
aux.interpolate(sol_ig - as_vector((x[0], x[1], 0)))
file.write(aux)
#sys.exit()

#Compute initial guess for the angle field
theta_ig = Function(W, name='IG theta')
# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
e_1 = Constant((1, 0))
e_2 = Constant((0, 1))
phi = pi/2
u_s = sqrt(3) * cos(phi/2)
v_s = 2 * sqrt( 2/ ( 5-3 * cos(phi) ) )
u_0 = u_s * e_1
v_0 = v_s * e_2
u_ts = sqrt(3) * cos( (theta_ig+phi)/2 )
v_ts = 2 * sqrt( 2/ ( 5-3 * cos(theta_ig+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )

#defining the energy to minimize
L = dot(grad(sol_ig).T, grad(sol_ig)) - dot(A_t.T, A_t)
c = 1 #coercivity constant
energy = inner(L, L) * dx + c * inner(grad(theta_ig), grad(theta_ig)) * dx
zeta = TestFunction(W)
a = derivative(energy, theta_ig, zeta)

#Solve
solve(a == 0, theta_ig, solver_parameters={'quadrature_degree': '2'}) #'snes_monitor': None, 'snes_max_it': 10})

#Output IG in theta
file = VTKFile("IG_theta.pvd")
file.write(theta_ig)

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
bcs = [DirichletBC(Z.sub(0), boundary_CL, 1), DirichletBC(Z.sub(0), boundary_CR, 2)]

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_ts = sqrt(3) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2/ ( 5-3 * cos(theta+phi) ) )
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
#c_1, c_2, d_1, d_2, d_3 = 5, .5, 1e-2, 1e-2, 1e-2 #problematic set
c_1, c_2, d_1, d_2, d_3 = 1.0, 0.5, 0.1, 0.1, 1e-2

#Total energy
dens = c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( H, H)
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
#a -=  inner( dot(avg(G), n('+')), jump(grad(w))) * dS # consistency term
en_pen = .5 * inner( dot(avg(G), n('+')), jump(grad(y))) * dS # consistency and symmetry energy term
a -= derivative(en_pen, y, w)
a += alpha / h_avg * inner( jump( grad(y), n ), jump( grad(w), n ) ) * dS #pen term

#Symmetry term
HH = variable(grad(grad(w)))
#GG = replace(G, {H:HH})
q = v_t_p * v_ts * inner( HH, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( HH,outer(N,v_0,v_0) )
dens = c_2 * q**2 + d_3 * inner(HH, HH)
GG = diff(dens, HH)
#a -=  inner( dot(avg(GG), n('+')), jump(grad(y))) * dS #symmetry term

#Gradient BC
a += alpha / h * inner( dot(grad(y), n), dot(grad(w), n) ) * ds #lhs pen
a -= alpha / h * inner( dot(grad(y_ref), n), dot(grad(w), n) ) * ds #rhs pen
a -=  inner( dot(dot(G, n), n), dot(grad(w), n)) * ds #consistency term
#a -= inner( dot(dot(GG, n), n), dot(grad(y), n)) * ds #lhs symmetry term
#a += inner( dot(dot(GG, n), n), dot(grad(y_ref), n)) * ds #rhs symmetry term

#Solve
#parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
parameters = {'snes_monitor': None, 'snes_max_it': 10, 'quadrature_degree': '2'}
solve(a == 0, sol, bcs=bcs, solver_parameters=parameters)


#plotting the results
aux = Function(V, name='yeff 3d')
aux.interpolate(sol.sub(0)-as_vector((x[0], x[1], 0)))
file = VTKFile('surf_comp.pvd')
file.write(aux)

aux = Function(W, name='theta')
aux.assign(sol.sub(1))
file = VTKFile('theta_comp.pvd')
file.write(aux)

#Computation of error
err_y = errornorm(y_ref, sol.sub(0), norm_type='H1')
PETSc.Sys.Print('Error in y: %.3e' % err_y)
err_theta = errornorm(theta_ref, sol.sub(1), norm_type='H1')
PETSc.Sys.Print('Error in theta: %.3e' % err_theta)

#Output error in y
file = VTKFile('err_disp.pvd')
aux = Function(W, name='err_disp')
u_ref = y_ref - as_vector((x[0], x[1], 0))
aux.interpolate(sqrt(inner(y - y_ref, y - y_ref)) / sqrt(inner(u_ref, u_ref)))
file.write(aux)

#Output error in theta
file = VTKFile('err_theta.pvd')
aux = Function(W, name='err_theta')
aux.interpolate(abs(theta - theta_ref))
file.write(aux)
