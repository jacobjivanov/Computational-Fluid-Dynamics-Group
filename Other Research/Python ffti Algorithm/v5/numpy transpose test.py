import numpy as np

Ni, Nj, Nk = 3, 4, 5
xyz = np.zeros(shape = (Ni, Nj, Nk))
for i in range(0, Ni):
   for j in range(0, Nj):
      for k in range(0, Nk):
         xyz[i, j, k] = 100 * i + 10 * j + 1 * k

print("xyz array:\n {0} \n".format(xyz))

yxz = np.transpose(xyz, axes = (1, 0, 2))

print("yxz array:\n {0}".format(yxz))