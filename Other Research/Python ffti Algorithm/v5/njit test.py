# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
from numpy.fft import fft
from numba import jit, njit, prange
import ffti_v5 as fi
from time import time

def jinter_1D(coords, values, pos):
   """
   returns the one dimensional FFT interpolated value at the given position
   
   Parameters: 
      coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array
         Example:
         [x0,  x1,  ..., xNi]
      values : ArrayLike
         a list or numpy.array of values associated with the coord_array
         Example:
         [v(x0), v(x1), ..., v(xNi)]
      pos : Float
         A float with the coordinate of the desired interpolated value
         Example: x
   """

   return inter_1D_sub(fft(values[:-1]), len(values) - 1, coords[-1], pos)

@jit(nopython = True)
def inter_1D_sub(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0]) / Ni
   for f in prange(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 / Ni * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 / Ni * np.imag(values_fft[f]) * np.sin(w * pos)

   return value_inter

x = np.linspace(0, 2 * np.pi, 100)
y = np.cos(x) - np.sin(2 * x)

x_inter = np.linspace(0, 2 * np.pi, 1000)
jy_inter = np.zeros(1000)
y_inter = np.zeros(1000)

t0 = time()
for i in prange(len(x_inter)):
   jy_inter[i] = jinter_1D(x, y, pos = x_inter[i])
print(time() - t0)

t0 = time()
for i in range(len(x_inter)):
   y_inter[i] = fi.inter_1D(x, y, pos = x_inter[i])
print(time() - t0)

import matplotlib.pyplot as plt
plt.scatter(x, y, color = 'red', label = "Discrete")
plt.plot(x_inter, jy_inter, color = 'blue', label = "Interpolated")
plt.legend()
plt.show()