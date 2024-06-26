import numpy as np

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
    return np.array([omegau_1, omegau_2, omegau_3]), np.array([omegav_1, omegav_2, omegav_3])

