# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import ffti_v6 as fi
import numpy as np
from time import time

t_init = time()

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 21, 23, 17

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
x_slice_inter = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_ana = np.zeros(shape = (Nj_inter, Nk_inter))
x_slice_error = np.zeros(shape = (Nj_inter, Nk_inter))
for j in range(0, Nj_inter):
   for k in range(0, Nk_inter):
      x_slice_inter[j, k] = fi.inter_3D(x, y, z, rho, pos = [x_pos, y_inter[j], z_inter[k]])
      x_slice_ana[j, k] = func(x_pos, y_inter[j], z_inter[k])
      x_slice_error[j, k] = x_slice_inter[j, k] - x_slice_ana[j, k]
      print("Interpolating x-slice. \tProgress: {0:07.3f}% Complete".format(100 * (j + ((k + 1 ) / Nk_inter) ) / Nj_inter), end = '\r')
t_x = time()
print("Interpolating x-slice. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_x - t_build))

y_pos = 2 * np.pi * np.random.rand()
y_slice_inter = np.zeros(shape = (Ni_inter, Nk_inter))
y_slice_ana = np.zeros(shape = (Ni_inter, Nk_inter))
y_slice_error = np.zeros(shape = (Ni_inter, Nk_inter))
for i in range(0, Ni_inter):
   for k in range(0, Nk_inter):
      y_slice_inter[i, k] = fi.inter_3D(x, y, z, rho, pos = [x_inter[i], y_pos, z_inter[k]])
      y_slice_ana[i, k] = func(x_inter[i], y_pos, z_inter[k])
      y_slice_error[i, k] = y_slice_inter[i, k] - y_slice_ana[i, k]
      print("Interpolating y-slice. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((k + 1 ) / Nk_inter) ) / Ni_inter), end = '\r')
t_y = time()
print("Interpolating y-slice. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_y - t_x))

z_pos = 2 * np.pi * np.random.rand()
z_slice_inter = np.zeros(shape = (Ni_inter, Nj_inter))
z_slice_ana = np.zeros(shape = (Ni_inter, Nj_inter))
z_slice_error = np.zeros(shape = (Ni_inter, Nj_inter))
for i in range(0, Ni_inter):
   for j in range(0, Nj_inter):
      z_slice_inter[i, j] = fi.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_pos])
      z_slice_ana[i, j] = func(x_inter[i], y_inter[j], z_pos)
      z_slice_error[i, j] = z_slice_inter[i, j] - z_slice_ana[i, j]
      print("Interpolating z-slice. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
t_z = time()
print("Interpolating z-slice. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_z - t_y))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 3, figsize = (15, 10), constrained_layout = True)

fig.suptitle("Function: " + r"$\rho(x, y, z) = e^{\sin(x) + \sin(y) + \sin(z)}$" + ", Ni = {0}, Nj = {1}, Nk = {2}".format(Ni, Nj, Nk))

yz = np.meshgrid(y_inter, z_inter, indexing = 'ij')
ax[0][0].set_title("x = {0:.4f} Analytical Slice".format(x_pos))
cont_x_ana = ax[0][0].contourf(yz[0], yz[1], x_slice_ana, cmap = 'bwr')
fig.colorbar(cont_x_ana, ax = ax[0][0])
ax[0][1].set_title("x = {0:.4f} Interpolation Slice".format(x_pos))
cont_x_inter = ax[0][1].contourf(yz[0], yz[1], x_slice_inter, cmap = 'bwr')
fig.colorbar(cont_x_inter, ax = ax[0][1])
ax[0][2].set_title("x = {0:.4f} Interpolation Error".format(x_pos))
cont_x_error = ax[0][2].contourf(yz[0], yz[1], x_slice_error, cmap = 'bwr', levels = 100)
fig.colorbar(cont_x_error, ax = ax[0][2])
for n in [0, 1, 2]: ax[0][n].set(xlabel = "y-axis", ylabel = "z-axis")

xz = np.meshgrid(x_inter, z_inter, indexing = 'ij')
ax[1][0].set_title("y = {0:.4f} Analytical Slice".format(y_pos))
cont_y_ana = ax[1][0].contourf(xz[0], xz[1], y_slice_ana, cmap = 'bwr')
fig.colorbar(cont_y_ana, ax = ax[1][0])
ax[1][1].set_title("y = {0:.4f} Interpolation Slice".format(y_pos))
cont_y_inter = ax[1][1].contourf(xz[0], xz[1], y_slice_inter, cmap = 'bwr')
fig.colorbar(cont_y_inter, ax = ax[1][1])
ax[1][2].set_title("y = {0:.4f} Interpolation Error".format(y_pos))
cont_y_error = ax[1][2].contourf(xz[0], xz[1], y_slice_error, cmap = 'bwr', levels = 100)
fig.colorbar(cont_y_error, ax = ax[1][2])
for n in [0, 1, 2]: ax[1][n].set(xlabel = "x-axis", ylabel = "z-axis")

xy = np.meshgrid(x_inter, y_inter, indexing = 'ij')
ax[2][0].set_title("z = {0:.4f} Analytical Slice".format(z_pos))
cont_z_ana = ax[2][0].contourf(xy[0], xy[1], z_slice_ana, cmap = 'bwr')
fig.colorbar(cont_z_ana, ax = ax[2][0])
ax[2][1].set_title("z = {0:.4f} Interpolation Slice".format(z_pos))
cont_z_inter = ax[2][1].contourf(xy[0], xy[1], z_slice_inter, cmap = 'bwr')
fig.colorbar(cont_z_inter, ax = ax[2][1])
ax[2][2].set_title("z = {0:.4f} Interpolation Error".format(z_pos))
cont_z_error = ax[2][2].contourf(xy[0], xy[1], z_slice_error, cmap = 'bwr', levels = 100)
fig.colorbar(cont_z_error, ax = ax[2][2])
for n in [0, 1, 2]: ax[2][n].set(xlabel = "x-axis", ylabel = "y-axis")

fig.savefig("test_3D slices v5.png", dpi = 600)
plt.show()
