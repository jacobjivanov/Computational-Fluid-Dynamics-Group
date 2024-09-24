def ga(Ni):
   import ffti_v6 as fi6
   import ffti_v7 as fi7
   import li_v6 as li6
   import numpy as np
   import numba

   @numba.njit
   def func(x, y, z):
      rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
      return rho
   
   Nj, Nk = 12, 11
   x_max = 2 * np.pi, 
   y_max, z_max = x_max, x_max
   x = np.linspace(0, x_max, Ni)
   y, z = np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

   rho = np.zeros(shape = (Ni, Nj, Nk))
   for i in range(Ni):
      for j in range(Nj):
         for k in range(Nk):
            rho[i, j, k] = func(x[i], y[j], z[k])
   # np.save("Convergence Test Outputs/rho[Ni = {}].npy".format(Ni), rho)
   
   Ni_inter = 4 * Ni
   Nj_inter, Nk_inter = 4 * Nj, 4 * Nk

   x_inter = np.linspace(0, x_max, Ni_inter)
   y_inter, z_inter = np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)
   
   rho_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
   rho_fi7 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

   for i in range(Ni_inter):
      for j in range(Nj_inter):
         for k in range(Nk_inter):
            rho_fi6[i, j, k] = fi6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
            rho_fi7[i, j, k] = fi7.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])

         print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
   print("Interpolating. \tProgress: 100.000% Complete")
   
   np.save("fi7 testing/rho_fi6[Ni = {}].npy".format(Ni), rho_fi6)
   np.save("fi7 testing/rho_fi7[Ni = {}].npy".format(Ni), rho_fi7)

ga(6)

"""
>>> import numpy as np
>>> rho_fi6 = np.load("rho_fi6[Ni = 6].npy")
>>> rho_fi7 = np.load("rho_fi7[Ni = 6].npy")
>>> diff = rho_fi7 - rho_fi6
>>> diff = np.ravel(diff)
>>> print(np.mean(diff))
0.0
>>> print(np.std(diff))
0.0
"""