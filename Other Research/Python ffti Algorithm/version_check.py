import numpy as np
import numba
import v6.ffti_v6 as fi6
import v7.ffti_v7 as fi7

Ni, Nj, Nk = 11, 11, 11
x_max = 2 * np.pi, 
y_max, z_max = x_max, x_max
x = np.linspace(0, x_max, Ni)
y = np.linspace(0, y_max, Nj)
z = np.linspace(0, z_max, Nk)

@numba.njit
def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
   return rho

rho = np.zeros(shape = (Ni, Nj, Nk))
for i in range(Ni):
   for j in range(Nj):
      for k in range(Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])

Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
x_inter = np.linspace(0, x_max, Ni_inter)
y_inter = np.linspace(0, y_max, Nj_inter)
z_inter = np.linspace(0, z_max, Nk_inter)

rho_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
error_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

for i in range(Ni_inter):
   for j in range(Nj_inter):
      for k in range(Nk_inter):
         rho_fi6[i, j, k] = fi6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
         error_fi6[i, j, k] = rho_fi6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])

      print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')