from firedrake import *
import sys

# Create mesh
L = 10
H = 10
size_ref = 25 #25 #10 #debug
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Load ref computation from a file
ref = Function(Z, name='ref')
y_ref, theta_ref = ref.sub(0), ref.sub(1)

#Define the boundary conditions
bcs = [DirichletBC(Z.sub(0), y_ref, 1), DirichletBC(Z.sub(0), y_ref, 2), DirichletBC(Z.sub(0), y_ref, 3), DirichletBC(Z.sub(0), y_ref, 4), DirichletBC(Z.sub(1), theta_ref, 1), DirichletBC(Z.sub(1), theta_ref, 2), DirichletBC(Z.sub(1), theta_ref, 3), DirichletBC(Z.sub(1), theta_ref, 4)]

#Define trial and test functions
V = TestFunction(Z)
w, eta = split(V)
U = Function(Z)
y, theta = split(U)

#Define solutions
#sol = Function(Z, name='sol')
#y, theta = sol.sub(0), sol.sub(1)

# elastic parameters
c_1, c_2, d_1, d_2, d_3 = 1.0, 1.0, 1e-2, 1e-2, 1e-2

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
e_1 = Constant((1, 0))
e_2 = Constant((0, 1))
phi = pi/6
u_s = sqrt(3.0) * cos(phi/2)
v_s = 2 * sqrt( 2.0/ ( 5-3 * cos(phi) ) )
u_0 = u_s * e_1
v_0 = v_s * e_2
u_ts = sqrt(3.0) * cos( (theta+phi)/2 )
v_ts = 2 * sqrt( 2.0/ ( 5-3 * cos(theta+phi) ) )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, variable(theta))
v_t_p = diff(v_ts, variable(theta))

#Preparation for variational form
H = variable( grad(grad(y)) )
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - A_t.T * A_t
q = v_t_p * v_ts * inner( H, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( H,outer(N,v_0,v_0) )

#Total energy
Energy = (c_1 * inner( L, L ) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner( grad(theta), grad(theta) ) + d_3 * inner( N, N )) * dx

# first variation of the energy
a = derivative(Energy, U, V)
