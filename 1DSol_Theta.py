import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ## ODE solve -- actuation field(\theta)
def actuation(vec, x, phi, c_tau, c_kappa):
    theta, d_theta = vec
    lu = 2.0 * np.sqrt( 2/(5 - 3 * np.cos( theta+phi )) )
    lv = np.sqrt(3.0) * np.cos( (theta+phi)/2.0 )
    lpu = - 3 * np.sqrt(2) * np.sin(theta+phi)/(5 - 3 * np.cos( theta+phi ))**(3/2)
    lpv = - np.sqrt(3) * np.sin( (theta+phi)/2 ) / 2
    d = ( 3 * np.cos( 3*(theta+phi)/2 ) - 5 * np.cos( (theta+phi)/2 ) ) / 8 / np.sqrt( 10/3 - 2 * np.cos( theta+phi ) )
    equ = [d_theta, - d * lu/lpv * d_theta**2 + c_tau**2 * lu**2/lpv/lv**3 + c_kappa**2 * lpu/lu/lpv**2]
    return equ


# initial conditions for \theta, \tau and \kappa, range
phi = np.pi/6
c_tau = 0.0
c_kappa = -0.02907*6
theta0 = [0.2 - np.pi/6, 2.0]
x = np.linspace(0, 10, 101)

sol_theta = odeint(actuation, theta0, x, args=(phi, c_tau, c_kappa))

plt.figure(1)
plt.plot(x, sol_theta[:, 0], 'b', label=r'$\theta(x)$')
plt.plot(x, sol_theta[:, 1], 'g', label=r"$\theta^{'}(x)$")
plt.legend(loc='best')
plt.xlabel(r'$x$')
plt.grid()
# plt.axis('equal')
# plt.axis('scaled')
plt.xlim((0, 10))
# plt.ylim((-0.5, 0.8))
# plt.show()

# solutions to bend and twist fields
def bendtwist(vec, phi, c_tau, c_kappa):
    theta, d_theta = vec[:, 0], vec[:, 1]
    # lu0 = 2.0 * np.sqrt(2 / (5 - 3 * np.cos(phi)))
    lu = 2.0 * np.sqrt(2 / (5 - 3 * np.cos(theta + phi)))
    lv = np.sqrt(3) * np.cos((theta + phi) / 2)
    lpu = - 3 * np.sqrt(2) * np.sin(theta + phi) / (5 - 3 * np.cos(theta + phi)) ** (3 / 2)
    lpv = - np.sqrt(3) * np.sin((theta + phi) / 2) / 2
    kappa = c_kappa / lv/lu**2/lpv
    tau = c_tau / lv**2
    omegau_1 = tau * lu
    omegau_2 = kappa * lu * lpu * lv
    omegau_3 = 0.0 * lu
    omegav_1 = kappa * lpv * lv * lu
    omegav_2 = - tau * lv
    omegav_3 = d_theta * lpv/lu
    return [omegau_1, omegau_2, omegau_3], [omegav_1, omegav_2, omegav_3]


omega_u, omega_v = bendtwist(sol_theta, phi, c_tau, c_kappa)

# plt.figure(2)
# plt.plot(x, omega_v[2], 'b', label=r"$\kappa(x)$")
# plt.legend(loc='best')
# plt.xlabel(r'$x$')
# plt.grid()
# plt.xlim((0, 10))
# plt.show()

# ## test of matrix-vector product
aaa = np.dot(np.ones((3, 3)), np.array(omega_u))

plt.figure(2)
plt.plot(x, aaa[0], 'b', label=r"$\kappa(x)$")
plt.legend(loc='best')
plt.xlabel(r'$x$')
plt.grid()
plt.xlim((0, 10))
plt.show()
# ############
