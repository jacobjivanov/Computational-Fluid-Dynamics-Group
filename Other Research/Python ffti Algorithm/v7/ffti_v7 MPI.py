# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# ffti_v7 makes a few major changes from ffti_v6:
# 1. Removes ordering from inter_3D(), as it provides no benefit to interpolation accuracy but makes library much less readable.
# 2. Has built in fft capability, not relying on numpy, in order to have numba.njit functionality in that regard.
# 3. Has parallelization built in.

import numpy as np
from numpy.fft import fft
import numba
from mpi4py import MPI
import dask.array as da

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()
HELPER_SIZE = SIZE - 1

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k
# fourier term indexes: f

def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter = value_inter +  2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter = value_inter - 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
   value_inter /= Ni
   return value_inter

def inter_1D(coords, values, pos):
   # see "ffti_v7 reference.txt" for function docstring
   return reconstruction(fft(values[:-1]), len(values) - 1, coords[-1], pos)

def inter_3D(x_coords, y_coords, z_coords, values3D, pos):
   # see "ffti_v7 reference.txt" for function docstring

   Ni, Nj, Nk = values3D.shape
   
   # Ni = len(values3D)
   # Nj = len(values3D[0])
   # Nk = len(values3D[0][0])

   if RANK == 0:
      inter_zy_values = np.zeros(Ni)
      r = 0
      while TBD < TBD:
         comm.Send(values3D[i][j], dest = r, tag = 14)
         comm.Recv(inter_y_values[j], source = (r + 1) % SIZE, tag = 14)



      for i in range(0, Ni):
         inter_y_values = np.zeros(Nj)
         for j in range(0, Nj):
            comm.Send(values3D[i][j], dest = TBD, tag = 14)
            comm.Recv(inter_y_values[j], source = TBD, tag = 14)

         inter_zy_values[i] = inter_1D(y_coords, inter_y_values, pos[1])

      inter_zyx_value = inter_1D(x_coords, inter_zy_values, pos[0])
      return inter_zyx_value
   
   if RANK != 0:
      TBD