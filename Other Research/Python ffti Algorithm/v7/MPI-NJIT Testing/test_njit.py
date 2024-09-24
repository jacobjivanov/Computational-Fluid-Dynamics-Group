import numpy as np
from numpy.fft import fft
from numba import jit, njit, prange

@njit
def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   
   # fourier term indexes: f
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
   value_inter /= Ni
   return value_inter 

def inter_1D(coords, values, pos):
   return reconstruction(fft(values[:-1]), len(values) - 1, coords[-1], pos)

x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x)

import ffti_v7 as fi7
pos = 1.2
a = fi7.inter_1D(x, y, pos)
print(a)