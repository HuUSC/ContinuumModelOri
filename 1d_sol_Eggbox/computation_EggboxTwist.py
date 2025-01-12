from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc

# Create mesh
u_s = sqrt(2.0)
v_s = sqrt(2.0)
e_1 = Constant((1, 0))
e_2 = Constant((0, 1))
u_0 = u_s * e_1
v_0 = v_s * e_2
L = u_s
H = v_s
size_ref = 20 #80 computation #10 #debug
mesh = RectangleMesh(size_ref, size_ref, L, H, diagonal='crossed')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
x = SpatialCoordinate(mesh)
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

#Load ref computation from a file
with CheckpointFile("Ref.h5", 'r') as afile:
    meshRef = afile.load_mesh("meshRef")
    Y_ref = afile.load_function(meshRef, "yeff")
    Theta_ref = afile.load_function(meshRef, "theta")
ref = Function(Z, name='ref')
y_ref, theta_ref = ref.sub(0), ref.sub(1)
theta_ref.interpolate(Theta_ref)
y_ref.sub(0).interpolate(Y_ref[0])
y_ref.sub(1).interpolate(Y_ref[1])
y_ref.sub(2).interpolate(Y_ref[2])

# characteristic out-of-plane displacement
disp_out = np.linalg.norm( y_ref.at(L/2, H/2) - (y_ref.at(0, 0) + y_ref.at(L, H))/2 )
PETSc.Sys.Print('Characteristic out-of-plane displacement: %.3e' % disp_out)
# sys.exit()

#Estimating max norms
# u_inf = norm(y_ref - as_vector((x[0], x[1], 0)), 'l200')
# print(u_inf)
# theta_inf = norm(theta_ref, 'l200')
# print(theta_inf)
# sys.exit()

#Interior penalty
h = CellDiameter(mesh) # cell diameter
h_avg = avg(h)  # average size of cells sharing a facet
n = FacetNormal(mesh) # outward-facing normal vector

#Nonlinear problem
#Define trial and test functions
test = TestFunction(Z)
w, eta = split(test)

#Define solutions
sol = Function(Z, name='sol')
y, theta = split(sol)

#Interpolate initial guess
sol.sub(0).interpolate(y_ref)
#Go get code from Hu for the computation of theta
sol.sub(1).interpolate(theta_ref)

#Define the boundary conditions
bcs = [DirichletBC(Z.sub(0), y_ref, 1), DirichletBC(Z.sub(0), y_ref, 2),
       DirichletBC(Z.sub(0), y_ref, 3), DirichletBC(Z.sub(0), y_ref, 4),
       DirichletBC(Z.sub(1), theta_ref, 1), DirichletBC(Z.sub(1), theta_ref, 2),
       DirichletBC(Z.sub(1), theta_ref, 3), DirichletBC(Z.sub(1), theta_ref, 4)]

# basis vectors & reference/deformed Bravais lattice vectors & metric tensor
u_ts = 2 * sin( ( acos( 1-cos( variable(theta) ) ) - variable(theta) )/2 )
v_ts = 2 * sin( ( acos( 1-cos( variable(theta) ) ) + variable(theta) )/2 )
A_t = as_matrix( [ [ u_ts/ u_s, 0], [0, v_ts/v_s] ] )
u_t_p = diff(u_ts, variable(theta)) #variable(theta))
v_t_p = diff(v_ts, variable(theta)) #variable(theta))

#Preparation for variational form
H = variable( grad(grad(y)) )
N = cross(y.dx(0), y.dx(1))
N /= sqrt(inner(N, N))
L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)
q = v_t_p * v_ts * inner( H, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( H,outer(N,v_0,v_0) )
J = sqrt(det(dot(grad(y).T, grad(y))))

# elastic parameters
# c_1, c_2, d_1, d_2, d_3 = 1.0, 0.5, 0, 0, 2e-2 # working
c_1, c_2, d_1, d_2, d_3 = 1.0, 0.5, 1e-3, 0, 0  # working

#Total energy
dens = (c_1 * inner( L, L ) / J  + c_2 * q**2/( (u_s*v_s)**4 ) + d_1 * inner( H, H)
        + d_2 * inner( grad(theta), grad(theta) ) + d_3 * theta**2)
G = diff(dens, H)
Energy = dens * dx

# first variation of the energy
a = derivative(Energy, sol, test)

# interior penalty
alpha = Constant(0.1)
a -=  inner( dot(avg(G), n('+')), jump(grad(w))) * dS # consistency term
#en_pen = inner( dot(avg(G), n('+')), jump(grad(y))) * dS # consistency and symmetry energy term
#a -= derivative(en_pen, y, w)
a += alpha / h_avg * inner(jump(grad(y)), jump(grad(w))) * dS #pen term

#Gradient BC
a += alpha / h * inner( dot(grad(y), n), dot(grad(w), n) ) * ds #lhs pen
a -= alpha / h * inner( dot(grad(y_ref), n), dot(grad(w), n) ) * ds #rhs pen
a -=  inner( dot(dot(G, n), n), dot(grad(w), n)) * ds #consistency term
#a -= inner( dot(dot(GG, n), n), dot(grad(y), n)) * ds #lhs symmetry term
#a += inner( dot(dot(GG, n), n), dot(grad(y_ref), n)) * ds #rhs symmetry term

#Solve
#parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
parameters = {'snes_monitor': None, 'snes_max_it': 20, 'quadrature_degree': '4', 'rtol': 1e-8, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
solve(a == 0, sol, bcs=bcs, solver_parameters=parameters)


#plotting the results
surface = Function(V, name='yeff 3d')
surface.interpolate(sol.sub(0)-as_vector((x[0], x[1], 0)))
# file = VTKFile('surf_comp.pvd')
# file.write(aux)

ang = Function(W, name='theta')
ang.assign(sol.sub(1))
file = VTKFile('effective_comp.pvd')
file.write(surface, ang)

# sys.exit()

#Computation of error
err_y = errornorm(y_ref, sol.sub(0), norm_type='H1')
PETSc.Sys.Print('Error in y: %.3e' % err_y)
err_theta = errornorm(theta_ref, sol.sub(1), norm_type='H1')
PETSc.Sys.Print('Error in theta: %.3e' % err_theta)

#Output error in y
file = VTKFile('err_norm.pvd')
y_err = Function(W, name='err_disp')
# u_ref = y_ref - as_vector((x[0], x[1], 0))
y_err.interpolate(sqrt(inner(y - y_ref, y - y_ref))) #/ u_inf) #sqrt(inner(u_ref, u_ref)))
# file.write(aux)

#Output error in theta
# file = VTKFile('err_theta.pvd')
theta_err = Function(W, name='err_theta')
#aux.interpolate(abs(theta - theta_ref))
theta_err.interpolate(abs(theta - theta_ref)) # / theta_inf) #abs(theta_ref + phi))
file.write(y_err, theta_err)

#Output all errors
# print(assemble(c_1 * inner(L, L) * dx))
# print(assemble(c_2 * q**2 * dx))
# #print(assemble( d_1 * theta**2 * dx(mesh)))
# #print(assemble(d_2 * inner( grad(theta), grad(theta) ) * dx(mesh)))
# print(assemble(d_3 * inner( H, H) * dx))
