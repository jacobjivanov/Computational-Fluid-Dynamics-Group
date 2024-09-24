# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import matplotlib.pyplot as plt
import numpy as np
import FFTW_interpolate as fi

x_max = 10
y_max = 10

Ni = 11
Nj = 11

x, y = np.meshgrid(np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj))

z = np.zeros(shape = (Nj, Ni))
for j in range(0, Nj):
   for i in range(0, Ni):
      z[j][i] = np.e ** (np.sin(x[0][i]) + np.sin(y[j][0]))
      # z[j][i] = np.e ** np.sin(x[0][i]) + np.log(y[j][0] + 1)
      # z[j][i] = x[0][i] - y[j][0] ** 1.2

Ni_inter = 51
Nj_inter = 51

x_inter, y_inter = np.meshgrid(np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter))

z_approx = np.zeros(shape = (Nj_inter, Ni_inter))
for j in range(0, Nj_inter):
   for i in range(0, Ni_inter):
      # z_approx[j][i] = fi.inter_2D(x, y, z, [i * x_max / Ni_inter, j * y_max / Nj_inter])
      z_approx[j][i] = fi.inter_2D(x, y, z, [x_inter[0][i], y_inter[j][0]])

import matplotlib.pyplot as plt

plt.scatter(x[0], z[0], color = 'grey', label = 'discrete input')
plt.plot(x_inter[0], z_approx[0], color = 'blue', label = 'continuous interpolation')
plt.legend(loc = 'upper center')
plt.show()