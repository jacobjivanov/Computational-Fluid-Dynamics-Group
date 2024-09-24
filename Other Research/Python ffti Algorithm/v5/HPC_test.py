import numpy as np
import li_v5 as li

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 10, 10, 10

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

rho = np.zeros(shape = (Ni, Nj, Nk))

def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))

for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])
      print("Building Data. \tProgress: {0:07.3f}% Complete".format(100 * (i + (j + 1) / Nj) / Ni), end = '\r')
print("Building Data. \tProgress: 100.000% Complete")


Ni_inter, Nj_inter, Nk_inter = Ni * 4, Nj * 4, Nk * 4
x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)


rho_inter = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
for i in range(0, Ni_inter): 
   for j in range(0, Nj_inter): 
      for k in range(0, Nk_inter):
         rho_inter[i, j, k] = li.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
      print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (i + (j + 1) / Nj_inter) / Ni_inter), end = '\r')
print("Interpolating. \tProgress: 100.000% Complete")

import matplotlib.pyplot as plt
plt.contour(np.meshgrid(x, y, indexing = 'ij')[0], np.meshgrid(x, y, indexing = 'ij')[1], rho[5])
# plt.contour(np.meshgrid(x_inter, y_inter, indexing = 'ij')[0], np.meshgrid(x_inter, y_inter, indexing = 'ij')[1], rho_inter[10])
plt.show()