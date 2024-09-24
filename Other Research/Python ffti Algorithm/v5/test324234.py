import numpy as np
from time import time

Ni, Nj, Nk = 11, 11, 11

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
   return rho

t_init = time()
rho = np.zeros(shape = (Ni, Nj, Nk))
for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])

def norm(values3D, Ni, Nj, Nk):
   sum = 0
   for i in range(Ni):
      for j in range(Nj):
         for k in range(Nk):
            sum += values3D[i, j, k] ** 2
   return sum ** 0.5 * Ni * Nj * Nk

print(norm(rho, Ni, Nj, Nk))