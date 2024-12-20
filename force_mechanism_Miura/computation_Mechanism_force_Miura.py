import numpy as np
from firedrake import *
import sys
from firedrake.output import *
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt

# Create mesh
N = 30 #80 computation #10 #debug
mesh = UnitSquareMesh(N, N, diagonal='crossed')
# mesh = Mesh('mesh.msh')

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
PETSc.Sys.Print('Nb dof: %i' % Z.dim())

step = 0.01
max_i = int(2.61/step + 1)
i = 1
disp_force = [[0.0, 0.0]]

while i < max_i:
    phi = pi / 6.0
    theta_0 = step*i
    u_s_0 = sqrt(3.0) * cos(phi / 2.0)
    v_s_0 = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(phi)))
    u_ts_0 = sqrt(3.0) * cos((theta_0 + phi) / 2.0)
    v_ts_0 = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(theta_0 + phi)))

    # Define the boundary conditions
    x = SpatialCoordinate(mesh)
    # BC
    boundary_CL = as_vector((u_ts_0 / u_s_0 * x[0], v_ts_0 / v_s_0 * x[1]))
    boundary_CR = as_vector((u_ts_0 / u_s_0 * x[0], v_ts_0 / v_s_0 * x[1]))
    boundary_CT = as_vector((u_ts_0 / u_s_0 * x[0], v_ts_0 / v_s_0 * x[1]))
    boundary_CB = as_vector((u_ts_0 / u_s_0 * x[0], v_ts_0 / v_s_0 * x[1]))

    bcs = [DirichletBC(V, boundary_CL, 1), DirichletBC(V, boundary_CR, 2),
           DirichletBC(V, boundary_CT, 3), DirichletBC(V, boundary_CB, 4)]

    # Interior penalty
    alpha = Constant(1e-1)  # 1e2 #10 #penalty parameter
    h = CellDiameter(mesh)  # cell diameter
    h_avg = avg(h)  # average size of cells sharing a facet
    n = FacetNormal(mesh)  # outward-facing normal vector

    # Initial guess
    # Bilinear form
    # u = TrialFunction(V)
    # v = TestFunction(V)
    # a = inner(grad(grad(u)), grad(grad(v))) * dx \
    #     - inner(dot(avg(grad(grad(u))), n('+')), jump(grad(v))) * dS \
    #     - inner(jump(grad(u)), dot(avg(grad(grad(v))), n('+'))) * dS \
    #     + 8.0 / h_avg * inner(jump(grad(u)), jump(grad(v))) * dS

    # Penalty term for the gradient Dirichlet bc
    # a += alpha/h * inner(dot(grad(u), n), dot(grad(v), n)) * (ds(1) + ds(2))

    # Lhs boundary penalty term
    # a -= inner(dot(grad(u), n), dot(dot(grad(grad(v)), n), n)) * (ds(1) + ds(2)) + inner(dot(grad(v), n), dot(dot(grad(grad(u)), n), n)) * (ds(1) + ds(2))

    # Linear form
    # L = Constant(0) * v[0] * dx

    # Solve variational problem
    sol_ig = Function(V, name='IG')
    v_basis = VectorSpaceBasis(constant=True, comm=COMM_WORLD)
    # solve(a == L, sol_ig, bcs, nullspace=v_basis, solver_parameters={'quadrature_degree': '3'})

    # test
    sol_ig.interpolate(as_vector((u_ts_0 / u_s_0 * x[0], v_ts_0 / v_s_0 * x[1])))

    # Save solution to file
    # file = VTKFile("IG.pvd")
    # aux = Function(V, name='IG')
    # # x = SpatialCoordinate(mesh)
    # aux.interpolate(sol_ig - as_vector((x[0], x[1])))
    # file.write(aux)
    # sys.exit()

    # Compute initial guess for the angle field
    theta_ig = Function(W, name='IG theta')
    # basis vectors & reference/deformed Bravais lattice vectors & metric tensor
    # e_1 = Constant((1, 0))
    # e_2 = Constant((0, 1))
    # u_s = sqrt(3) * cos(phi / 2)
    # v_s = 2 * sqrt(2 / (5 - 3 * cos(phi)))
    # u_0 = u_s * e_1
    # v_0 = v_s * e_2
    # u_ts = sqrt(3) * cos((theta_ig + phi) / 2)
    # v_ts = 2 * sqrt(2 / (5 - 3 * cos(theta_ig + phi)))
    # A_t = as_matrix([[u_ts / u_s, 0], [0, v_ts / v_s]])

    # elastic parameters
    # c_1, c_2, d_1, d_2, d_3 = 5, .5, 1e-2, 1e-2, 1e-2
    # c_1, c_2, d_1, d_2, d_3 = 1.0, 0.5, 0.1, 0.1, 1e-2 #old problematic set
    c_1 = 5.0  # metric constraint
    d_1, d_2= 1e-2, 1e-2
    d_3 = 1e-1

    # defining the energy to minimize
    # L = dot(grad(sol_ig).T, grad(sol_ig)) - dot(A_t.T, A_t)
    # energy = c_1 * inner(L, L) * dx + d_2 * inner(grad(theta_ig), grad(theta_ig)) * dx
    # zeta = TestFunction(W)
    # a = derivative(energy, theta_ig, zeta)

    # Solve
    # solve(a == 0, theta_ig, solver_parameters={'quadrature_degree': '2'})  # 'snes_monitor': None, 'snes_max_it': 10})
    # solve(a == 0, theta_ig)
    theta_ig.interpolate(theta_0)

    # Output IG in theta
    # file = VTKFile("IG_theta.pvd")
    # file.write(theta_ig)

    # PETSc.Sys.Print('Initial guess ok!\n')

    # Nonlinear problem
    # Define trial and test functions
    test = TestFunction(Z)
    w, eta = split(test)

    # Define solutions
    sol = Function(Z, name='sol')
    y, theta = split(sol)

    # Interpolate initial guess
    sol.sub(0).interpolate(sol_ig)
    # Go get code from Hu for the computation of theta
    sol.sub(1).interpolate(theta_ig)

    # Define the boundary conditions
    bcs_ = [DirichletBC(Z.sub(0), boundary_CL, 1), DirichletBC(Z.sub(0), boundary_CR, 2),
            DirichletBC(Z.sub(0), boundary_CT, 3), DirichletBC(Z.sub(0), boundary_CB, 4)]
    # bcs_ = [DirichletBC(Z.sub(1), Constant(theta_0), 1), DirichletBC(Z.sub(1), Constant(theta_0), 2),
    #         DirichletBC(Z.sub(1), Constant(theta_0), 3), DirichletBC(Z.sub(1), Constant(theta_0), 4)]


    # basis vectors & reference/deformed Bravais lattice vectors & metric tensor
    u_s = sqrt(3.0) * cos(phi / 2.0)
    v_s = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(phi)))
    u_ts = sqrt(3.0) * cos((theta + phi) / 2.0)
    v_ts = 2.0 * sqrt(2.0 / (5.0 - 3.0 * cos(theta + phi)))
    A_t = as_matrix([[u_ts / u_s, 0.0], [0.0, v_ts / v_s]])
    # u_t_p = diff(u_ts, sol)
    # v_t_p = diff(v_ts, sol)

    # Preparation for variational form
    H = variable(grad(grad(y)))
    # N = cross(y.dx(0), y.dx(1))
    # N /= sqrt(inner(N, N))
    L = dot(grad(y).T, grad(y)) - dot(A_t.T, A_t)
    J = sqrt(det(dot(grad(y).T, grad(y))))
    # q = v_t_p * v_ts * inner( H, outer(N,u_0,u_0)  ) + u_t_p * u_ts * inner( H,outer(N,v_0,v_0) )

    # Test
    # en = assemble(c_1 * inner(L, L) * dx)
    # print(en)
    # #en = assemble(c_2 * q**2 * dx)
    # #print(en)
    # en = assemble(d_1 * theta**2 * dx)
    # print(en)
    # en = assemble(d_2 * inner(grad(theta), grad(theta)) * dx)
    # print(en)
    # en = assemble( d_3 * inner(H, H) * dx)
    # print(en)

    # Total energy
    # c_2 = 0 #test
    # dens = c_1 * inner(L, L) + c_2 * q**2 + d_1 * theta**2 + d_2 * inner(grad(theta), grad(theta)) + d_3 * inner(H, H)
    dens = c_1 * inner(L, L)/J + d_3 * theta ** 2 + d_2 * inner(grad(theta), grad(theta)) + d_1 * inner(H, H)  # test
    G = diff(dens, H)
    # G  = 2 * d_3 * H
    # G_ = 2 * d_3 * variable(grad(grad(w)))
    Energy = dens * dx

    # first variation of the energy
    a = derivative(Energy, sol, test)

    # interior penalty
    # a -= inner(dot(avg(G), n('+')), jump(grad(w))) * dS  # consistency term
    # a += alpha / h_avg * inner(jump(grad(y)), jump(grad(w))) * dS  # pen term

    # #Gradient BC
    # a += alpha / h * inner(dot(grad(y), n), dot(grad(w), n)) * (ds(1) + ds(2)) #lhs pen
    # en_pen = inner(dot(dot(G, n), n), dot(grad(y), n)) * (ds(1) + ds(2)) # consistency and symmetry energy term
    # a -= derivative(en_pen, y, w)

    # Solve
    # parameters={"snes_monitor": None, "ksp_type": "preonly", "mat_type": "aij", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    parameters = {'snes_max_it': 50, 'quadrature_degree': '4'}  # , "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    nullspace = MixedVectorSpaceBasis(Z, [v_basis, Z.sub(1)])
    try:
        # solve(a == 0, sol, bcs=bcs_, nullspace=nullspace, solver_parameters=parameters)
        solve(a == 0, sol, bcs=bcs_, nullspace=nullspace)
    except ConvergenceError:  # firedrake.exceptions.ConvergenceError:
        en = assemble(c_1 * inner(L, L) * dx)
        print(en)
        #    en = assemble(c_2 * q**2 * dx)
        #    print(en)
        en = assemble(d_1 * theta ** 2 * dx)
        print(en)
        en = assemble(d_2 * inner(grad(theta), grad(theta)) * dx)
        print(en)
        en = assemble(d_3 * inner(H, H) * dx)
        print(en)
        sys.exit()

    # plotting the results
    # aux = Function(V, name='yeff 2d')
    # aux.interpolate(sol.sub(0)-as_vector((x[0], x[1])))
    # file = VTKFile('surf_comp.pvd')
    # file.write(aux)
    #
    # aux = Function(W, name='theta')
    # aux.assign(sol.sub(1))
    # file = VTKFile('theta_comp.pvd')
    # file.write(aux)

    # Computing reaction forces
    v_reac = Function(Z)
    bc_l = DirichletBC(V.sub(0), Constant(1), 1)
    bc_l.apply(v_reac.sub(0))
    res_l = assemble(action(a, v_reac))
    # print('Reaction on the left: %.3e' % res_l)
    # sys.exit()
    v_reac.sub(0).interpolate(Constant((0, 0)))
    bc_r = DirichletBC(V.sub(0), Constant(1), 2)
    bc_r.apply(v_reac.sub(0))
    res_r = assemble(action(a, v_reac))
    # print('Reaction on the right: %.3e' % res_r)
    # PETSc.Sys.Print('Total force: %.3e' % (res_l + res_r))
    print(f"Initial angle {theta_0: .2f}: Reaction on the left = {res_l: .3e}")

    disp_ = 1.0 - u_ts_0 / u_s_0 * 1.0
    disp_force.append([disp_, res_l])
    i += 1


disp = Function(V, name='yeff 2d')
disp.interpolate(sol.sub(0) - as_vector((x[0], x[1])))
ang = Function(W, name='theta')
ang.assign(sol.sub(1))

outfile = VTKFile("output.pvd")
outfile.write(disp, ang)

# plt.plot(np.arange(i)*0.05, force)
disp_force_array = np.array(disp_force)
np.savetxt('DispForce.csv', disp_force_array, delimiter=',')
plt.plot(disp_force_array[:, 0], disp_force_array[:, 1])
plt.show()
