# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import ffti_v6 as fi6
import li_v6 as li6
import numpy as np
from numba import njit
from time import time

@njit
def l2norm_3D(values3D, Ni, Nj, Nk):
   sum = 0
   for i in range(Ni):
      for j in range(Nj):
         for k in range(Nk):
            sum += values3D[i, j, k]
   return sum ** 0.5

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

D = np.linspace(3, 5, 3) ** 2

li6_error = np.zeros(6), np.zeros(6) 
fi6_error = np.zeros(6), np.zeros(6)

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi

for d in range(len(D)):
   Ni, Nj, Nk = int(D[d]), int(D[d]), int(D[d])
   print("Ni = {0}".format(Ni))
   x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

   rho = generate_rho(Ni, Nj, Nk)
   
   Ni_inter, Nj_inter, Nk_inter = 4 * Ni, 4 * Nj, 4 * Nk
   x_inter, y_inter, z_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter), np.linspace(0, z_max, Nk_inter)

   rho_li6 = generate_rho_li6(Ni_inter, Nj_inter, Nk_inter)
   error_li6 = error_li6(Ni_inter, Nj_inter, Nk_inter)
   rho_fi6 = generate_rho_fi6(Ni_inter, Nj_inter, Nk_inter)
   error_fi6 = error_fi6(Ni_inter, Nj_inter, Nk_inter)

"""
import matplotlib.pyplot as plt
plt.scatter(D, li6_error[0], label = "li6 Error Average")
plt.errorbar(D, li6_error[0], yerr = li6_error[1], label = "li6 Error Standard Deviation")
plt.scatter(D, fi6_error[0], label = "fi6 Error Average")
plt.errorbar(D, fi6_error[0], yerr = fi6_error[1], label = "fi6 Error Standard Deviation")
plt.show()

plt.scatter(D, li6_error[0], label = "li6 Error Average")
plt.errorbar(D, li6_error[0], yerr = li6_error[1], label = "li6 Error Standard Deviation")
plt.scatter(D, fi6_error[0], label = "fi6 Error Average")
plt.errorbar(D, fi6_error[0], yerr = fi6_error[1], label = "fi6 Error Standard Deviation")
"""