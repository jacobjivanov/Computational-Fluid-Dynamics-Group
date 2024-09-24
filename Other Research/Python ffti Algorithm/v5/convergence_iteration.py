# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

def ci(Ni):
   # convergence_iteration
   import ffti_v6 as fi6
   import li_v6 as li6
   import numpy as np
   from numba import njit
   from time import time

   @njit
   def l2norm_3D(values3D, Ni, Nj, Nk):
      x_max = 2 * np.pi
      sum = 0
      for i in range(Ni):
         for j in range(Nj):
            for k in range(Nk):
               sum += np.abs(values3D[i, j, k] ** 2 * x_max  ** 3 / ((Ni - 1) * (Nj - 1) * (Nk - 1)))
               # print(sum)
      return sum ** 0.5
   
   # def l2norm_3D(values_subset, n, Ni):
   #    x_max = 2 * np.pi
   #    sum = 0
   #    for i in range(n):
   #       sum += np.abs( * x_max  ** 3 / ((Ni - 1) * (Nj - 1) * (Nk - 1)))
   #       # print(sum)
   #    return sum ** 0.5

   @njit
   def func(x, y, z):
      rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
      return rho

   def generate_rho(Ni, Nj, Nk):
      rho = np.zeros(shape = (Ni, Nj, Nk))
      for i in range(0, Ni): 
         for j in range(0, Nj): 
            for k in range(0, Nk):
               rho[i, j, k] = func(x[i], y[j], z[k])
         print("rho Generation. \tProgress: {0:07.3f}% Complete".format(100 * i / Ni), end = '\r')
      print("rho Generation. \tProgress: 100.000% Complete")

      return rho

   def generate_rho_li6(Ni_inter, Nj_inter, Nk_inter):
      rho_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
      for i in range(0, Ni_inter): 
         for j in range(0, Nj_inter): 
            for k in range(0, Nk_inter):
               rho_li6[i, j, k] = li6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]], order = 'zyx')
         print("li6 Interpolation. \tProgress: {0:07.3f}% Complete".format(100 * i / Ni_inter), end = '\r')
      print("li6 Interpolation. \tProgress: 100.000% Complete")

      return rho_li6

   def generate_rho_fi6(Ni_inter, Nj_inter, Nk_inter):
      rho_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
      for i in range(0, Ni_inter): 
         for j in range(0, Nj_inter): 
            for k in range(0, Nk_inter):
               rho_fi6[i, j, k] = fi6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]], order = 'zyx')
         print("fi6 Interpolation. \tProgress: {0:07.3f}% Complete".format(100 * i / Ni_inter), end = '\r')
      print("fi6 Interpolation. \tProgress: 100.000% Complete")


      return rho_fi6

   def error_li6(Ni_inter, Nj_inter, Nk_inter):
      error_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
      for i in range(0, Ni_inter): 
         for j in range(0, Nj_inter): 
            for k in range(0, Nk_inter):
               error_li6[i, j, k] = rho_li6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])
         print("li6 Error. \t\tProgress: {0:07.3f}% Complete".format(100 * i / Ni_inter), end = '\r')
      print("li6 Error. \t\tProgress: 100.000% Complete")
      
      return error_li6

   def error_fi6(Ni_inter, Nj_inter, Nk_inter):
      error_fi6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter))
      for i in range(0, Ni_inter): 
         for j in range(0, Nj_inter): 
            for k in range(0, Nk_inter):
               error_fi6[i, j, k] = rho_fi6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])
         print("fi6 Error. \t\tProgress: {0:07.3f}% Complete".format(100 * i / Ni_inter), end = '\r')
      print("fi6 Error. \t\tProgress: 100.000% Complete")

      return error_fi6

   x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi

   Ni = Ni
   Nj, Nk = Ni, Ni
   print("Ni = {0}".format(Ni))
   x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

   rho = generate_rho(Ni, Nj, Nk)

   Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
   x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)

   rho_li6 = generate_rho_li6(Ni_inter, Nj_inter, Nk_inter)
   error_li6 = error_li6(Ni_inter, Nj_inter, Nk_inter)
   print(l2norm_3D(error_li6, Ni_inter, Nj_inter, Nk_inter))
   rho_fi6 = generate_rho_fi6(Ni_inter, Nj_inter, Nk_inter)
   error_fi6 = error_fi6(Ni_inter, Nj_inter, Nk_inter)
   print(l2norm_3D(error_fi6, Ni_inter, Nj_inter, Nk_inter))


# def cif(Ni):
#    # convergence_iteration_fast
#    import ffti_v6 as fi6
#    import li_v6 as li6
#    import numpy as np
#    from numba import njit
#    from time import time
#    from random import random

#    @njit
#    def l2norm_3D(values3D, Ni, Nj, Nk):
#       x_max = 2 * np.pi
#       sum = 0
#       for i in range(Ni):
#          for j in range(Nj):
#             for k in range(Nk):
#                sum += np.abs(values3D[i, j, k] * x_max  ** 3 / ((Ni - 1) * (Nj - 1) * (Nk - 1)))
#                # print(sum)
#       return sum ** 0.5

#    @njit
#    def func(x, y, z):
#       rho = np.e ** (np.sin(x) + np.sin(y) + np.sin(z))
#       return rho

#    def generate_rho(Ni, Nj, Nk):
#       rho = np.zeros(shape = (Ni, Nj, Nk))
#       for i in range(0, Ni): 
#          for j in range(0, Nj): 
#             for k in range(0, Nk):
#                rho[i, j, k] = func(x[i], y[j], z[k])
#          print("rho Generation. \tProgress: {0:07.3f}% Complete".format(100 * i / Ni), end = '\r')
#       print("rho Generation. \tProgress: 100.000% Complete")

#       return rho

#    def error_li6_subset(Ni_inter, Nj_inter, Nk_inter):
#       error_li6_subset = np.zeros(500)
      
#       for r in range(0, 500):
#          p = [x_max * random(), y_max * random(), z_max * random()]
#          error_li6_subset[r] = li6.inter_3D(x, y, z, rho, pos = [*p], order = 'zyx') - func(*p)
#          print("li6 Error Subset. \tProgress: {0:07.3f}% Complete".format(100 * r / 500), end = '\r')
#       print("li6 Error Subset. \tProgress: 100.000% Complete")
      
#       return error_li6_subset

#    def error_fi6_subset(Ni_inter, Nj_inter, Nk_inter):
#       error_fi6_subset = np.zeros(500)
      
#       for r in range(0, 500):
#          p = [x_max * random(), y_max * random(), z_max * random()]
#          error_fi6_subset[r] = fi6.inter_3D(x, y, z, rho, pos = [*p], order = 'zyx') - func(*p)
#          print("fi6 Error Subset. \tProgress: {0:07.3f}% Complete".format(100 * r / 500), end = '\r')
#       print("fi6 Error Subset. \tProgress: 100.000% Complete")
      
#       return error_fi6_subset

#    x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi

#    Ni = Ni
#    Nj, Nk = Ni, Ni
#    print("Ni = {0}".format(Ni))
#    x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

#    rho = generate_rho(Ni, Nj, Nk)

#    Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
#    x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)

#    rho_li6 = generate_rho_li6(Ni_inter, Nj_inter, Nk_inter)
#    error_li6_subset = error_li6_subset(Ni_inter, Nj_inter, Nk_inter)

#    print(l2norm_3D(error_li6_subset, Ni_inter, Nj_inter, Nk_inter))
   
#    rho_fi6 = generate_rho_fi6(Ni_inter, Nj_inter, Nk_inter)
#    error_fi6_subset = error_fi6_subset(Ni_inter, Nj_inter, Nk_inter)


#    print(l2norm_3D(error_fi6, Ni_inter, Nj_inter, Nk_inter))