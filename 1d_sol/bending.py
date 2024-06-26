from firedrake import *
from firedrake.petsc import PETSc
import sys
sys.path.append('.')
from ode_solve import *
from scipy.integrate import odeint
from firedrake.output import VTKFile

# initial conditions for \theta, \tau and \kappa
phi = np.pi/6
c_tau = 0.0
c_kappa = -0.02907*6
theta0 = [0.2 - np.pi/6, 2.0]

#time-step
N = 100
t = np.linspace(0, 10, N)

#Solving the ode to get \theta and \omega
sol_theta = odeint(actuation, theta0, t, args=(phi, c_tau, c_kappa))
sol_omega_u, sol_omega_v = bendtwist(sol_theta, phi, c_tau, c_kappa)
#print(sol_omega_v)
#sys.exit()

# Create mesh
L = 10
H = 10
size_ref = 25 #25 #10 #debug
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')

# Define function space
V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#Interpolate results from the ode computation
theta = Function(V, name='theta')
coord = Function(V)
x = SpatialCoordinate(mesh)
coord.interpolate(x[0])
coords = coord.vector().array()
theta.vector()[:] = np.interp(coords, t, sol_theta[:,0])

#plotting the result for theta
final = VTKFile('theta.pvd')
final.write(theta)

#Interpolate results from the ode computation
W = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
PETSc.Sys.Print('Nb vec dof: %i' % W.dim())
omega_u = Function(W, name='omega_u')
omega_u.vector()[:,1] = np.interp(coords, t, sol_omega_u[1,:])
omega_v = Function(W, name='omega_v')
omega_v.vector()[:,0] = np.interp(coords, t, sol_omega_v[0,:])
omega_v.vector()[:,2] = np.interp(coords, t, sol_omega_v[2,:])

#plotting the result for omega
final = VTKFile('omega_u.pvd')
final.write(omega_u)
final = VTKFile('omega_v.pvd')
final.write(omega_v)

#Computing the rotation
Z = TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3))
PETSc.Sys.Print('Nb tensor dof: %i' % Z.dim())

#Dirichlet BC
bcs = [DirichletBC(Z, Identity(3), 1)]

#Auxiliary fields
Omega_u = as_tensor(((0, -omega_u[2], omega_u[1]), (omega_u[2], 0, -omega_u[0]), (-omega_u[1], omega_u[0], 0)))
Omega_v = as_tensor(((0, -omega_v[2], omega_v[1]), (omega_v[2], 0, -omega_v[0]), (-omega_v[1], omega_v[0], 0)))

#Bilinear form
S = TrialFunction(Z)
T = TestFunction(Z)
a = inner(S.dx(1) - dot(S, Omega_v), T.dx(1) - dot(T, Omega_v)) * dx + inner(S.dx(0) - dot(S, Omega_u), T.dx(0) - dot(T, Omega_u)) * dx
l = Constant(0) * T[0,0] * dx(mesh)

#Linear solve
Reff = Function(Z, name='rotation')
solve(a == l, Reff, bcs=bcs)

#plotting the result
final = VTKFile('rot.pvd')
final.write(Reff)

#Recomputing the effective deformation

# the coefficient functions
lv = sqrt(3) * cos(0.5*(theta+phi))
lu = 2 * sqrt ( 2 / (5-3*cos(theta+phi)))


#Bilinear form
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u), grad(v)) * dx
Aeff = as_tensor(((lu, 0), (0, lv), (0, 0)))
l = inner(dot(Reff, Aeff), grad(v)) * dx

#Linear solve
yeff = Function(W, name='yeff')
nullspace = VectorSpaceBasis(constant=True)
solve(a == l, yeff, nullspace=nullspace)

#plotting the result
aux = Function(W, name='yeff 3d')
aux.interpolate(yeff-as_vector((x[0], x[1], 0)))
final = VTKFile('yeff.pvd')
final.write(aux)
#final.write(yeff)

