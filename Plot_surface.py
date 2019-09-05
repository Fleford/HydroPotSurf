import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load surface
surface = np.loadtxt("h_field.txt")
# zmin = np.min(surface[np.nonzero(surface)])
zmin = 0
zmax = surface.max()
print(zmin)
print(zmax)
print(surface)

# Plot 3d Surface
fig = plt.figure()
ax = Axes3D(fig, azim=-90.0, elev=90.0)

surface[surface == 0] = np.nan
# surface = surface
Z = surface
X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
ax.plot_surface(X, -Y, Z)
ax.set_zlim3d(zmin, zmax)
plt.show()
