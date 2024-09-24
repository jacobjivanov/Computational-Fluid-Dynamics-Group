import ffti_v6 as fi
import numpy as np
from time import time

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 21, 21, 21

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

for i in range(len(x)):
   print(x[i])

rho = np.zeros(shape = (Ni, Nj, Nk))

def func(x, y, z):
   rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
   return rho

for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])
         print("rho[{0}][{1}][{2}] = {3}".format(i, j, k, rho[i, j, k]))