from firedrake import *
import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf

#Load ref computation from a file
with CheckpointFile("Errors_Eggbox.h5", 'r') as afile:
    mesh = afile.load_mesh('mesh')
    err_disp = afile.load_function(mesh, "err_disp")
    err_theta = afile.load_function(mesh, "err_theta")

#plot
#fig, axes = plt.subplots()
#levels = np.linspace(0, 1, 51)
contours = tricontourf(err_theta)
#axes.set_aspect("equal")
plt.colorbar(contours)
plt.savefig('err_theta_EB.pdf')
plt.show()
