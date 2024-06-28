import numpy as np

# ## ODE solve -- actuation field(\theta)
def actuation(vec, x, phi, c_tau, c_kappa):
    theta, d_theta = vec
    lv = 2.0 * np.sqrt( 2/(5 - 3 * np.cos( theta+phi )) )
    lu = np.sqrt(3.0) * np.cos( (theta+phi)/2.0 )
    lpv = - 3 * np.sqrt(2) * np.sin(theta+phi)/(5 - 3 * np.cos( theta+phi ))**(3/2)
    lpu = - np.sqrt(3) * np.sin( (theta+phi)/2 ) / 2
    d = ( 3 * np.cos( 3*(theta+phi)/2 ) - 5 * np.cos( (theta+phi)/2 ) ) / 8 / np.sqrt( 10/3 - 2 * np.cos( theta+phi ) )
    equ = [d_theta, - d * lv/lpu * d_theta**2 + c_tau**2 * lv**2/lpu/lu**3 + c_kappa**2 * lpv/lv/lpu**2]
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
    omegau_3 = -d_theta * lpu / lv #0.0 * lu #correct?
    omegav_1 = kappa * lpv * lv * lu
    omegav_2 = - tau * lv
    omegav_3 = d_theta * lpv/lu
    return np.array([omegau_1, omegau_2, omegau_3]), np.array([omegav_1, omegav_2, omegav_3])

