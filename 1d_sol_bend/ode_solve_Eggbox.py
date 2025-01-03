import numpy as np
import sympy as sp

s = sp.symbols('s')
lu = 2 * sp.sin( ( sp.acos( 1-sp.cos(s) ) + s)/2 )
lv = 2 * sp.sin( ( sp.acos( 1-sp.cos(s) ) - s)/2 )
lpu = sp.diff(lu, s)
lpv = sp.diff(lv, s)

# ## ODE solve -- actuation field(\theta)
def actuation(vec, x, phi, c_tau, c_kappa):
    theta, d_theta = vec
    d = sp.diff(lpu/lv, s) * lv/lpu
    d = sp.lambdify(s, d, 'numpy')
    aux_1 = sp.lambdify(s, lv**2/lpu/lu**3, 'numpy')
    aux_2 = sp.lambdify(s, lpv/lv/lpu**2, 'numpy')
    equ = [d_theta, - d(theta) * d_theta**2 + c_tau**2 * aux_1(theta) + c_kappa**2 * aux_2(theta)]
    return equ

# solutions to bend and twist fields
def bendtwist(vec, phi, c_tau, c_kappa):
    theta, d_theta = vec[:, 0], vec[:, 1]
    kappa = c_kappa * sp.lambdify(s, 1/lv**2/lu/lpu, 'numpy')(theta)
    tau = c_tau / sp.lambdify(s, 1/lu**2, 'numpy')(theta)
    omegau_1 = tau * sp.lambdify(s, lu, 'numpy')(theta)
    omegau_2 = kappa * sp.lambdify(s, lu * lpu * lv, 'numpy')(theta)
    omegau_3 = -d_theta * sp.lambdify(s, lpu / lv, 'numpy')(theta)
    omegav_1 = kappa * sp.lambdify(s, lpv * lv * lu, 'numpy')(theta)
    omegav_2 = -tau * sp.lambdify(s, lv, 'numpy')(theta)
    omegav_3 = 0.0 * theta
    return np.array([omegau_1, omegau_2, omegau_3]), np.array([omegav_1, omegav_2, omegav_3])

#ODE for down BC
def ode_rot_d(t, R, omega_u, phi):
    aux = omega_u.at(t,0)
    Omega_u = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_u)
    return res.flatten()
    
#ODE for top BC
def ode_rot_t(t, R, omega_u, phi, H):
    aux = omega_u.at(t, H)
    Omega_u = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_u)
    return res.flatten()

#ODE for left BC
def ode_rot_l(t, R, omega_v, phi):
    aux = omega_v.at(0, t)
    Omega_v = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_v)
    return res.flatten()

#ODE for right BC
def ode_rot_r(t, R, omega_v, phi, L):
    aux = omega_v.at(L, t)
    Omega_v = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_v)
    return res.flatten()
