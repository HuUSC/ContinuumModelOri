import numpy as np
import sympy as sp

# ## ODE solve -- actuation field(\theta)
def actuation(vec, x, phi, c_tau, c_kappa):
    theta, d_theta = vec
    s = sp.symbols('s')
    lv = 2 * sp.sqrt( 2/(5 - 3 * sp.cos(s)) )
    lu = sp.sqrt(3.0) * sp.cos(.5*s )
    lpu = sp.diff(lu, s)
    lpv = sp.diff(lv, s)
    d = sp.diff(lpu/lv, s) * lv/lpu
    d = sp.lambdify(s, d, 'numpy')
    aux_1 = sp.lambdify(s, lv**2/lpu/lu**3, 'numpy')
    aux_2 = sp.lambdify(s, lpv/lv/lpu**2, 'numpy')
    equ = [d_theta, - d(theta+phi) * d_theta**2 + c_tau**2 * aux_1(theta+phi) + c_kappa**2 * aux_2(theta+phi)]
    #lv = 2.0 * np.sqrt( 2/(5 - 3 * np.cos( theta+phi )) )
    #lu = np.sqrt(3.0) * np.cos( (theta+phi)/2.0 )
    #lpv = - 3 * np.sqrt(2) * np.sin(theta+phi)/(5 - 3 * np.cos( theta+phi ))**(3/2)
    #lpu = - np.sqrt(3) * np.sin( (theta+phi)/2 ) / 2
    #d = ( 3 * np.cos( 3*(theta+phi)/2 ) - 5 * np.cos( (theta+phi)/2 ) ) / 8 / np.sqrt( 10/3 - 2 * np.cos( theta+phi ) )
    #equ = [d_theta, - d * lv/lpu * d_theta**2 + c_tau**2 * lv**2/lpu/lu**3 + c_kappa**2 * lpv/lv/lpu**2]
    return equ

# solutions to bend and twist fields
def bendtwist(vec, phi, c_tau, c_kappa):
    theta, d_theta = vec[:, 0], vec[:, 1]
    lv = 2.0 * np.sqrt(2 / (5 - 3 * np.cos(theta + phi)))
    lu = np.sqrt(3) * np.cos((theta + phi) / 2)
    lpv = - 3 * np.sqrt(2) * np.sin(theta + phi) / (5 - 3 * np.cos(theta + phi)) ** (3 / 2)
    lpu = - np.sqrt(3) * np.sin((theta + phi) / 2) / 2
    kappa = c_kappa / lu/lv**2/lpu
    tau = c_tau / lu**2
    omegau_1 = tau * lu
    omegau_2 = kappa * lu * lpu * lv
    omegau_3 = -d_theta * lpu / lv
    omegav_1 = kappa * lpv * lv * lu
    omegav_2 = -tau * lv
    omegav_3 = 0 * lv #d_theta * lpv/lu
    return np.array([omegau_1, omegau_2, omegau_3]), np.array([omegav_1, omegav_2, omegav_3])

#ODE for down BC
def ode_rot_d(t, R, omega_u, phi):
    lu0 = np.sqrt(3) * np.cos((phi) / 2)
    aux = omega_u.at(t,0) #t*lu0
    Omega_u = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_u)
    return res.flatten()
    
#ODE for top BC
def ode_rot_t(t, R, omega_u, phi, H):
    lu0 = np.sqrt(3) * np.cos((phi) / 2)
    aux = omega_u.at(t, H) #t*lu0
    Omega_u = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_u)
    return res.flatten()

#ODE for left BC
def ode_rot_l(t, R, omega_v, phi):
    lv0 = 2 * np.sqrt(2 / (5 - 3 * np.cos(phi)))
    aux = omega_v.at(0, t) #t*lv0
    Omega_v = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_v)
    return res.flatten()

#ODE for right BC
def ode_rot_r(t, R, omega_v, phi, L):
    lv0 = 2 * np.sqrt(2 / (5 - 3 * np.cos(phi)))
    aux = omega_v.at(L, t) #t*lv0
    Omega_v = np.array(((0, -aux[2], aux[1]), (aux[2], 0, -aux[0]), (-aux[1], aux[0], 0)))
    Rot = R.reshape((3,3))
    res = np.dot(Rot, Omega_v)
    return res.flatten()
