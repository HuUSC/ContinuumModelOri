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
L = 10 * lu0
H = 10 * lv0 
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
sol_d = solve_ivp(ode_rot_d, [0, L/lu0], np.identity(3).flatten(), t_eval=t, args=(omega_u, phi)) #method='BDF'
bc_d = Function(Z, name='BC Rot down')
coord.interpolate(x[0] / lu0)
coords = coord.vector().array()
#interpolate each of the coordinates...
bc_d.vector()[:, 0, 0] = np.interp(coords, sol_d.t, sol_d.y[0,:])
bc_d.vector()[:, 0, 1] = np.interp(coords, sol_d.t, sol_d.y[1,:])
bc_d.vector()[:, 0, 2] = np.interp(coords, sol_d.t, sol_d.y[2,:])
bc_d.vector()[:, 1, 0] = np.interp(coords, sol_d.t, sol_d.y[3,:])
bc_d.vector()[:, 1, 1] = np.interp(coords, sol_d.t, sol_d.y[4,:])
bc_d.vector()[:, 1, 2] = np.interp(coords, sol_d.t, sol_d.y[5,:])
bc_d.vector()[:, 2, 0] = np.interp(coords, sol_d.t, sol_d.y[6,:])
bc_d.vector()[:, 2, 1] = np.interp(coords, sol_d.t, sol_d.y[7,:])
bc_d.vector()[:, 2, 2] = np.interp(coords, sol_d.t, sol_d.y[8,:])

sol_l = solve_ivp(ode_rot_l, [0, H/lv0], np.identity(3).flatten(), t_eval=t, args=(omega_v, phi)) #method='BDF'
bc_l = Function(Z, name='BC Rot left')
coord.interpolate(x[1] / lv0)
coords = coord.vector().array()
bc_l.vector()[:, 0, 0] = np.interp(coords, sol_l.t, sol_l.y[0,:])
bc_l.vector()[:, 0, 1] = np.interp(coords, sol_l.t, sol_l.y[1,:])
bc_l.vector()[:, 0, 2] = np.interp(coords, sol_l.t, sol_l.y[2,:])
bc_l.vector()[:, 1, 0] = np.interp(coords, sol_l.t, sol_l.y[3,:])
bc_l.vector()[:, 1, 1] = np.interp(coords, sol_l.t, sol_l.y[4,:])
bc_l.vector()[:, 1, 2] = np.interp(coords, sol_l.t, sol_l.y[5,:])
bc_l.vector()[:, 2, 0] = np.interp(coords, sol_l.t, sol_l.y[6,:])
bc_l.vector()[:, 2, 1] = np.interp(coords, sol_l.t, sol_l.y[7,:])
bc_l.vector()[:, 2, 2] = np.interp(coords, sol_l.t, sol_l.y[8,:])

sol_t = solve_ivp(ode_rot_t, [0, L/lu0], bc_l.at(0, H).flatten(), t_eval=t, args=(omega_u, phi, H)) #method='BDF'
bc_t = Function(Z, name='BC Rot top')
coord.interpolate(x[0] / lu0)
coords = coord.vector().array()
bc_t.vector()[:, 0, 0] = np.interp(coords, sol_t.t, sol_t.y[0,:])
bc_t.vector()[:, 0, 1] = np.interp(coords, sol_t.t, sol_t.y[1,:])
bc_t.vector()[:, 0, 2] = np.interp(coords, sol_t.t, sol_t.y[2,:])
bc_t.vector()[:, 1, 0] = np.interp(coords, sol_t.t, sol_t.y[3,:])
bc_t.vector()[:, 1, 1] = np.interp(coords, sol_t.t, sol_t.y[4,:])
bc_t.vector()[:, 1, 2] = np.interp(coords, sol_t.t, sol_t.y[5,:])
bc_t.vector()[:, 2, 0] = np.interp(coords, sol_t.t, sol_t.y[6,:])
bc_t.vector()[:, 2, 1] = np.interp(coords, sol_t.t, sol_t.y[7,:])
bc_t.vector()[:, 2, 2] = np.interp(coords, sol_t.t, sol_t.y[8,:])

sol_r = solve_ivp(ode_rot_r, [0, H/lv0], bc_d.at(L, 0).flatten(), t_eval=t, args=(omega_v, phi, L)) #method='BDF'
bc_r = Function(Z, name='BC Rot right')
coord.interpolate(x[1] / lv0)
coords = coord.vector().array()
bc_r.vector()[:, 0, 0] = np.interp(coords, sol_r.t, sol_r.y[0,:])
bc_r.vector()[:, 0, 1] = np.interp(coords, sol_r.t, sol_r.y[1,:])
bc_r.vector()[:, 0, 2] = np.interp(coords, sol_r.t, sol_r.y[2,:])
bc_r.vector()[:, 1, 0] = np.interp(coords, sol_r.t, sol_r.y[3,:])
bc_r.vector()[:, 1, 1] = np.interp(coords, sol_r.t, sol_r.y[4,:])
bc_r.vector()[:, 1, 2] = np.interp(coords, sol_r.t, sol_r.y[5,:])
bc_r.vector()[:, 2, 0] = np.interp(coords, sol_r.t, sol_r.y[6,:])
bc_r.vector()[:, 2, 1] = np.interp(coords, sol_r.t, sol_r.y[7,:])
bc_r.vector()[:, 2, 2] = np.interp(coords, sol_r.t, sol_r.y[8,:])

#Test
print(bc_t.at(0,H))
print(bc_l.at(0,H))
sys.exit()
print(bc_r.at(L, H))
print(bc_t.at(L,H))
sys.exit()

#plotting the result
#aux = Function(Z)
#aux.interpolate(dot(bc_R.T, bc_R))
final = VTKFile('rot_BC.pvd')
final.write(bc_l)
sys.exit()

#Dirichlet BC
bcs = [DirichletBC(Z, bc_d, 3), DirichletBC(Z, bc_t, 4), DirichletBC(Z, bc_l, 1), DirichletBC(Z, bc_r, 2)]
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
