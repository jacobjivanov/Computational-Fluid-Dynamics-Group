# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# All following data comes from Table A.1 of Fundamentals of Heat and Mass Transfer
# by Theodore L. Bergman, Adrienne S. Lavine
T_BeO = np.array([300, 400, 600, 800, 1000, 1200, 1500, 2000]) # K
k_BeO = np.array([272, 196, 111, 70, 47, 33, 21.5, 15]) # W / m K

T_Steel = np.array([100, 200, 300, 400, 600, 800, 1000, 1200, 1500]) # K
k_Steel = np.array([9.2, 12.6, 14.9, 16.6, 19.8, 22.6, 25.4, 28.0, 31.7]) # W / m K

def k_BeO_fit(T, a, b):
   return a * b ** T

def k_Steel_fit(T, a, b):
   return a * T + b

k_BeO_params, k_BeO_covars = curve_fit(k_BeO_fit, T_BeO, k_BeO)
k_Steel_params, k_Steel_covars = curve_fit(k_Steel_fit, T_Steel, k_Steel)

def k(x, T): # m, K
   if (0 <= x) & (x <= 0.010):
      return k_BeO_fit(T, k_BeO_params[0], k_BeO_params[1])
   elif (0.010 < x) & (x <= 0.031):
      return k_Steel_fit(T, k_Steel_params[0], k_Steel_params[1])
   else:
      print("Out of Bounds Error: x position")

##################################################################################

# T_BeO = np.array([300, 400, 600, 800, 1000, 1200, 1500, 2000]) # K
cP_BeO = np.array([1030, 1350, 1690, 1865, 1975, 2055, 2145, 2750])

# T_Steel = np.array([100, 200, 300, 400, 600, 800, 1000, 1200, 1500]) # K
cP_Steel = np.array([272, 402, 477, 515, 557, 582, 611, 640, 682])

def cP_BeO_fit(T, a, b):
   return a * T + b

def cP_Steel_fit(T, a, b):
   return a * T + b

cP_BeO_params, cP_BeO_covars = curve_fit(cP_BeO_fit, T_BeO, cP_BeO)
cP_Steel_params, cP_Steel_covars = curve_fit(cP_Steel_fit, T_Steel, cP_Steel)

def cP(x, T): # m, K
   if (0 <= x) & (x <= 0.010):
      return cP_BeO_fit(T, cP_BeO_params[0], cP_BeO_params[1])
   elif (0.010 < x) & (x <= 0.031):
      return cP_Steel_fit(T, cP_Steel_params[0], cP_Steel_params[1])
   else:
      print("Out of Bounds Error: x position")

##################################################################################

def rho(x, T): # m, K
   if (0 <= x) & (x <= 0.010):
      return 3000 # kg / m3
   elif (0.010 < x) & (x <= 0.031):
      return 7900 # kg / m3
   else:
      print("Out of Bounds Error: x position")

##################################################################################

def alpha(x, T): # m, K
   return k(x, T) / (rho(x, T) * cP(x, T))