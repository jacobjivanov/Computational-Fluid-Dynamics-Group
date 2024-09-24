# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import li_v5 as li
import numpy as np
from time import time

t_init = time()

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 20, 20, 20

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

rho = np.zeros(shape = (Ni, Nj, Nk))# dtype = 'float64')

def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
   # rho = x + y + z
   # rho = np.e * (np.sin(x) + np.cos(y) + np.cos(z))
   # rho = np.e * (np.sin(x) + np.cos(y) + np.sqrt(np.cos(z) + 1))
   # rho = np.e ** np.sin(x + y + z)
   return rho

for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])
      print("Building rho Data. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj) ) / Ni), end = '\r')

t_build = time()
print("Building rho Data. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)

x_pos = 2 * np.pi * np.random.rand()


x_slice_ana = np.zeros(shape = (Nj_inter, Nk_inter))

x_slice_inter_xyz = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_inter_xzy = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_inter_yxz = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_inter_yzx = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_inter_zxy = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_inter_zyx = np.zeros(shape = (Nj_inter, Nk_inter))

x_slice_error_xyz = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error_xzy = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error_yxz = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error_yzx = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error_zxy = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error_zyx = np.zeros(shape = (Nj_inter, Nk_inter))

for j in range(0, Nj_inter):
   for k in range(0, Nk_inter):
      x_slice_ana[j, k] = func(x_pos, y_inter[j], z_inter[k])

      x_slice_inter_xyz[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'xyz')
      x_slice_inter_xzy[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'xzy')
      x_slice_inter_yxz[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'yxz')
      x_slice_inter_yzx[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'yzx')
      x_slice_inter_zxy[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'zxy')
      x_slice_inter_zyx[j, k] = li.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]], order = 'zyx')

      x_slice_error_xyz[j, k] = x_slice_inter_xyz[j, k] - x_slice_ana[j, k]
      x_slice_error_xzy[j, k] = x_slice_inter_xzy[j, k] - x_slice_ana[j, k]
      x_slice_error_yxz[j, k] = x_slice_inter_yxz[j, k] - x_slice_ana[j, k]
      x_slice_error_yzx[j, k] = x_slice_inter_yzx[j, k] - x_slice_ana[j, k]
      x_slice_error_zxy[j, k] = x_slice_inter_zxy[j, k] - x_slice_ana[j, k]
      x_slice_error_zyx[j, k] = x_slice_inter_zyx[j, k] - x_slice_ana[j, k]

      print("Interpolating x-slice. \tProgress: {0:07.3f}% Complete".format(100 * (j + ((k + 1 ) / Nk_inter) ) / Nj_inter), end = '\r')
t_x = time()
print("Interpolating x-slice. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_x - t_build))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize = (16, 9), constrained_layout = True)

fig.suptitle("Function: " + r"$\rho(x, y, z) = e^{\sin(x) + \sin(y) + \sin(z)}$" + ", Ni = {0}, Nj = {1}, Nk = {2}".format(Ni, Nj, Nk))

yz = np.meshgrid(y_inter, z_inter, indexing = 'ij')
ax[0][0].set_title("x = {0:.4f} xyz Interpolation Error".format(x_pos))
ax[1][0].set_title("x = {0:.4f} xzy Interpolation Error".format(x_pos))
ax[0][1].set_title("x = {0:.4f} yxz Interpolation Error".format(x_pos))
ax[1][1].set_title("x = {0:.4f} yzx Interpolation Error".format(x_pos))
ax[0][2].set_title("x = {0:.4f} zxy Interpolation Error".format(x_pos))
ax[1][2].set_title("x = {0:.4f} zyx Interpolation Error".format(x_pos))

cont_x_error_xyz = ax[0][0].contourf(yz[0], yz[1], x_slice_error_xyz, cmap = 'bwr', levels = 100)
cont_x_error_xzy = ax[1][0].contourf(yz[0], yz[1], x_slice_error_xyz, cmap = 'bwr', levels = 100)
cont_x_error_yxz = ax[0][1].contourf(yz[0], yz[1], x_slice_error_yxz, cmap = 'bwr', levels = 100)
cont_x_error_yzx = ax[1][1].contourf(yz[0], yz[1], x_slice_error_yzx, cmap = 'bwr', levels = 100)
cont_x_error_zxy = ax[0][2].contourf(yz[0], yz[1], x_slice_error_zxy, cmap = 'bwr', levels = 100)
cont_x_error_zyx = ax[1][2].contourf(yz[0], yz[1], x_slice_error_zyx, cmap = 'bwr', levels = 100)

fig.colorbar(cont_x_error_xyz, ax = ax[0][0])
fig.colorbar(cont_x_error_xzy, ax = ax[1][0])
fig.colorbar(cont_x_error_yxz, ax = ax[0][1])
fig.colorbar(cont_x_error_yzx, ax = ax[1][1])
fig.colorbar(cont_x_error_zxy, ax = ax[0][2])
fig.colorbar(cont_x_error_zyx, ax = ax[1][2])

fig.savefig("li order test_3D slices.png", dpi = 600)
plt.show()