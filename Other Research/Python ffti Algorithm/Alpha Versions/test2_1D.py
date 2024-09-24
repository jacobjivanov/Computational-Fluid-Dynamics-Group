# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import matplotlib.pyplot as plt
import numpy as np
import FFTW_interpolate as fi


x_max = 10
y_max = 10

Ni = 21
Nj = 1

x, y = np.meshgrid(np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj))

z = np.zeros(shape = (Nj, Ni))
for j in range(0, Nj):
   for i in range(0, Ni):
      z[j][i] = np.e ** (np.sin(x[0][i]) + np.sin(y[j][0]))

z_approx = np.zeros(shape = (Nj, Ni))
for j in range(0, Nj):
   for i in range(0, Ni):
      z_approx[j][i] = fi.inter_1D(x[0], z[0], x[0][i])

print(list(x[0]))
print(list(z[0]))

plt.plot(x[0], z[0])
plt.plot(x[0], z_approx[0])
plt.show()