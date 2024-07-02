from firedrake import *
from firedrake.petsc import PETSc
import sys
sys.path.append('.')
from ode_solve import *
from scipy.integrate import odeint,solve_ivp
from firedrake.output import VTKFile

# initial conditions for \theta, \tau and \kappa
phi = np.pi/6
c_tau = 0.0
c_kappa = -0.02907*6
theta0 = [0.2 - np.pi/6, 2.0]

#time-stepping
N = 100 #1000
Tf = 10
t = np.linspace(0, Tf, N)

#Solving the ode to get \theta and \omega
sol_theta = odeint(actuation, theta0, t, args=(phi, c_tau, c_kappa))
sol_omega_u, sol_omega_v = bendtwist(sol_theta, phi, c_tau, c_kappa)
#print(sol_omega_v)
#sys.exit()

# Create mesh
lu0 = np.sqrt(3) * np.cos((phi) / 2)
lv0 = 2 * np.sqrt(2 / (5 - 3 * np.cos(phi)))
L = 10 #* lv0
H = 10 * lv0 
print(L*H)
size_ref = 100 #100
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed', name='meshRef')

# Define function space
V = FunctionSpace(mesh, "CG", 1)
PETSc.Sys.Print('Nb dof: %i' % V.dim())

#Interpolate results from the ode computation
theta = Function(V, name='theta')
coord = Function(V)
x = SpatialCoordinate(mesh)
coord.interpolate(x[1] / lv0)
coords = coord.vector().array()
theta.vector()[:] = np.interp(coords, t, sol_theta[:,0])

#plotting the result for theta
final = VTKFile('theta.pvd')
final.write(theta)

#Interpolate results from the ode computation
W = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
PETSc.Sys.Print('Nb vec dof: %i' % W.dim())
omega_u = Function(W, name='omega_u')
omega_u.vector()[:,0] = np.interp(coords, t, sol_omega_u[0,:])
omega_u.vector()[:,1] = np.interp(coords, t, sol_omega_u[1,:])
omega_u.vector()[:,2] = np.interp(coords, t, sol_omega_u[2,:])
omega_v = Function(W, name='omega_v')
omega_v.vector()[:,0] = np.interp(coords, t, sol_omega_v[0,:])
omega_v.vector()[:,1] = np.interp(coords, t, sol_omega_v[1,:])
omega_v.vector()[:,2] = np.interp(coords, t, sol_omega_v[2,:])

#plotting the result for omega
final = VTKFile('omega_u.pvd')
final.write(omega_u)
final = VTKFile('omega_v.pvd')
final.write(omega_v)


#Computing the rotation
Z = TensorFunctionSpace(mesh, 'CG', 1, shape=(3,3))
PETSc.Sys.Print('Nb tensor dof: %i' % Z.dim())

#Solving the ODE to have the BC for the rotation
#test = ode_rot(3, np.identity(3), omega_u, phi)
#print(test)
sol_R = solve_ivp(ode_rot, [0, L/lu0], np.identity(3).flatten(), t_eval=t/lu0, method='BDF', args=(omega_u, phi))
bc_R = Function(Z, name='BC Rot')
coord.interpolate(x[0] / lu0)
coords = coord.vector().array()
#interpolate each of the coordinates...
bc_R.vector()[:, 0, 0] = np.interp(coords, sol_R.t, sol_R.y[0,:])
bc_R.vector()[:, 0, 1] = np.interp(coords, sol_R.t, sol_R.y[1,:])
bc_R.vector()[:, 0, 2] = np.interp(coords, sol_R.t, sol_R.y[2,:])
bc_R.vector()[:, 1, 0] = np.interp(coords, sol_R.t, sol_R.y[3,:])
bc_R.vector()[:, 1, 1] = np.interp(coords, sol_R.t, sol_R.y[4,:])
bc_R.vector()[:, 1, 2] = np.interp(coords, sol_R.t, sol_R.y[5,:])
bc_R.vector()[:, 2, 0] = np.interp(coords, sol_R.t, sol_R.y[6,:])
bc_R.vector()[:, 2, 1] = np.interp(coords, sol_R.t, sol_R.y[7,:])
bc_R.vector()[:, 2, 2] = np.interp(coords, sol_R.t, sol_R.y[8,:])
aux = Function(Z)
aux.interpolate(dot(bc_R.T, bc_R))

#plotting the result
final = VTKFile('rot_BC.pvd')
final.write(aux)
sys.exit()

#Dirichlet BC
bcs = [DirichletBC(Z, Identity(3), 1), DirichletBC(Z, bc_R, 3)]
#Complete the BC here when done with computing it

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
#Reff.interpolate(Identity(3))
print(assemble(inner(Reff.T, Reff) * dx))
#sys.exit()

#plotting the result
final = VTKFile('rot.pvd')
final.write(Reff)

#Recomputing the effective deformation

# the coefficient functions
u_s = sqrt(3.0) * cos(phi/2)
v_s = 2 * sqrt( 2.0/ ( 5-3 * cos(phi) ) )
u_ts = sqrt(3.0) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2.0/ ( 5-3 * cos(theta+phi) ) )


#Bilinear form
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u), grad(v)) * dx
Aeff = as_tensor(((u_ts/u_s, 0), (0, v_ts/v_s), (0, 0)))
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

# Saving the result
with CheckpointFile("Ref.h5", 'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(yeff)
    afile.save_function(theta)
