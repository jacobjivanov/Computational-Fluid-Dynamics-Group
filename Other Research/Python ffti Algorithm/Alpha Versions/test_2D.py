# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import matplotlib.pyplot as plt
import numpy as np
import fourier_interpolate as fi
import linear_interpolate as li

x_max = 10
y_max = 10

Ni = 21 # typically a round number + 1,
Nj = 21 # such that the interstitial values are nice

x, y = np.meshgrid(np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj))

def func(x, y):
   z = np.e ** (np.sin(x) + np.sin(y))
   # z = np.e ** np.sin(x ** 2) + np.log(y + 1)
   # z = x - y ** 1.2
   # z = np.sin(x ** y) - (x * y * np.cos(y / (x + 1)))

   return z

z = np.zeros(shape = (Nj, Ni))
for j in range(0, Nj):
   for i in range(0, Ni):
      z[j][i] = func(x[0][i], y[j][0])
      print("Building Data. Progress: {0:.3f}% Complete".format(100 * (j + (i + 1) / Ni) / Nj), end = '\r')

Ni_inter = 101 # typically a round number + 1,
Nj_inter = 101 # such that the interstitial values are nice

x_inter, y_inter = np.meshgrid(np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter))

z_fi_approx = np.zeros(shape = (Nj_inter, Ni_inter))
z_fi_error = np.zeros(shape = (Nj_inter, Ni_inter))
z_fi_order_diff = np.zeros(shape = (Nj_inter, Ni_inter))

z_li_approx = np.zeros(shape = (Nj_inter, Ni_inter))
z_li_error = np.zeros(shape = (Nj_inter, Ni_inter))

for j in range(0, Nj_inter):
   for i in range(0, Ni_inter):
      z_fi_approx[j][i] = fi.inter_2D(x, y, z, [x_inter[0][i], y_inter[j][0]])
      # z_fi_error[j][i] = z_fi_approx[j][i] - func(x_inter[0][i], y_inter[j][0])
      z_fi_order_diff[j][i] = fi.inter_2D(x, y, z, [x_inter[0][i], y_inter[j][0]], order = 'xy') - fi.inter_2D(x, y, z, [x_inter[0][i], y_inter[j][0]], order = 'yx')

      # z_li_approx[j][i] = li.inter_2D(x, y, z, [x_inter[0][i], y_inter[j][0]])
      # z_li_error[j][i] = z_li_approx[j][i] - func(x_inter[0][i], y_inter[j][0])

      # The following print command increases computation time of this double-for loop by approximately 2.7%
      print("Interpolating. Progress: {0:.3f}% Complete".format(100 * (j + (i + 1) / Ni_inter) / Nj_inter), end = '\r')

# PLOT SETTINGS BELOW
import matplotlib.pyplot as plt

fig = plt.figure()
fig.tight_layout()
ax = fig.add_subplot(projection = '3d')

# Discrete Input Scatter PLot
# ax.scatter(x, y, z, color = "grey", alpha = 1, label = "discrete input")

# Fourier Interpretation Surface Plot
# ax.plot_surface(x_inter, y_inter, z_fi_approx, cmap = "RdBu", alpha = 0.8)

# Linear Interpretation Surface Plot
# ax.plot_surface(x_inter, y_inter, z_li_approx, cmap = "cool", alpha = 0.4)

# Fourier Interpretation Error Surface Plot
# ax.plot_surface(x_inter, y_inter, z_fi_error, cmap = "jet", alpha = 0.4)

# Linear Interpretation Error Surface Plot
# ax.plot_surface(x_inter, y_inter, z_li_error, cmap = "cool", alpha = 0.4)

# Fourier Interpretation Order Difference Surface Plot
ax.plot_surface(x_inter, y_inter, z_fi_order_diff, cmap = "cool", alpha = 0.4)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

# ax.legend()
# ax.set_aspect('equal')

ax.view_init(azim = 225, elev = 0)
plt.show()
