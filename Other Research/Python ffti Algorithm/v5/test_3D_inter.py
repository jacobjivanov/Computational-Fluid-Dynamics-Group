import ffti_v6 as fi
import numpy as np
from time import time

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 16, 16, 16

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

rho = np.zeros(shape = (Ni, Nj, Nk))

def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
   return rho

for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])
         # print("rho[{0:.5f}][{1:.5f}][{2:.5f}] = {3}".format(x[i], y[j], z[k], rho[i, j, k]))

Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)

a = fi.inter_3D(x, y, z, rho, [1.57, 1.99, 2.52])
b = func(1.57, 1.99, 2.52)
print("rho_int({0:.5f}, {1:.5f}, {2:.5f}) = {3}".format(1.57, 1.99, 2.52, a))
print("rho_ana({0:.5f}, {1:.5f}, {2:.5f}) = {3}".format(1.57, 1.99, 2.52, b))

"""
rho_inter = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

for i in range(0, Ni_inter): 
   for j in range(0, Nj_inter): 
      for k in range(0, Nk_inter):
         rho_inter[i, j, k] = fi.inter_3D(x, y, z, rho, [x_inter[i], y_inter[j], z_inter[k]])
         print("rho_inter[{0}][{1}][{2}] = {3}".format(i, j, k, rho_inter[i, j, k]))

rho_inter_error = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

for i in range(0, Ni_inter): 
   for j in range(0, Nj_inter): 
      for k in range(0, Nk_inter):
         rho_inter_error[i, j, k] = rho_inter[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])
         print("rho_inter_error[{0}][{1}][{2}] = {3}".format(i, j, k, rho_inter_error[i, j, k]))
"""