def ga(Ni):
   import ffti_v6 as fi6
   import li_v6 as li6
   import numpy as np
   import numba

   @numba.njit
   def func(x, y, z):
      rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
      return rho
   
   Nj, Nk = Ni, Ni
   x_max = 2 * np.pi, 
   y_max, z_max = x_max, x_max
   x = np.linspace(0, x_max, Ni)
   y, z = x, x

   rho = np.zeros(shape = (Ni, Nj, Nk))
   for i in range(Ni):
      for j in range(Nj):
         for k in range(Nk):
            rho[i, j, k] = func(x[i], y[j], z[k])
   
   np.save("Convergence Test Outputs/rho[Ni = {}].npy".format(Ni), rho)

   Ni_inter = 4 * Ni
   Nj_inter, Nk_inter = Ni_inter, Ni_inter

   x_inter = np.linspace(0, x_max, Ni_inter)
   y_inter, z_inter = x_inter.copy(), x_inter.copy()

   rho_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
   rho_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

   error_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
   error_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))

   for i in range(Ni_inter):
      for j in range(Nj_inter):
         for k in range(Nk_inter):
            rho_li6[i, j, k] = li6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
            a = fi6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
            rho_fi6[i, j, k] = a

            error_li6[i, j, k] = rho_li6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])
            error_fi6[i, j, k] = rho_fi6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])

         print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
   print("Interpolating. \tProgress: 100.000% Complete")
   
   np.save("Convergence Test Outputs/rho_li6[Ni = {}].npy".format(Ni), rho_li6)
   np.save("Convergence Test Outputs/rho_fi6[Ni = {}].npy".format(Ni), rho_fi6)
   np.save("Convergence Test Outputs/error_li6[Ni = {}].npy".format(Ni), error_li6)
   np.save("Convergence Test Outputs/error_fi6[Ni = {}].npy".format(Ni), error_fi6)

ga(8)