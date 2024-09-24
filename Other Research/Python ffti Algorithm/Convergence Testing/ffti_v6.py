# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# THIS WAS A DIRECT COPY FROM THE v5 FOLDER

import numpy as np
from numpy.fft import fft
from numba import jit, njit, prange

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k
# fourier term indexes: f

# @njit(float64(float64))))
# @njit
def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter = value_inter +  2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter = value_inter - 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
   value_inter /= Ni
   return value_inter 

def inter_1D(coords, values, pos):
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
   
   return reconstruction(fft(values[:-1]), len(values) - 1, coords[-1], pos)

# @jit(parallel = True)
def slicing(d1_coords, d2_coords, d3_coords, Nd1, Nd2, Nd3, values3D, pos_d1, pos_d2, pos_d3):
   inter_d1d2_values = np.zeros(Nd3)

   for d3 in prange(0, Nd3):
      inter_d2_values = np.zeros(Nd2)
      for d2 in prange(0, Nd2):
         inter_d2_values[d2] = inter_1D(d1_coords, values3D[d3][d2], pos_d1)

      inter_d1d2_values[d3] = inter_1D(d2_coords, inter_d2_values, pos_d2)

   inter_d1d2d3_value = inter_1D(d3_coords, inter_d1d2_values, pos_d3)
   return inter_d1d2d3_value

def inter_3D(x_coords, y_coords, z_coords, values3D, pos, order = 'zyx'):
   """
   returns the two dimensional FFT interpolated value at the given position
   
   Parameters: 
      x_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [x0, x1, ..., xNi]
      y_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [y0, y1, ..., yNj]
      z_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [z0, z1, ..., zNk]
      
      values3D : ArrayLike
         a numpy.array of values associated with the coord_array with ij indexing
         Example:
         [  [  [v(x0, y0, z0), v(x0, y0, z1), ..., v(x0, y0, zNk)]
               [v(x0, y1, z0), v(x0, y1, z1), ..., v(x0, y1, zNk)]
               ...
               [v(x0, yNj, z0), v(x0, yNj, z1), ..., v(x0, yNj, zNk)]  ]

            [  [v(x1, y0, z0), v(x1, y0, z1), ..., v(x1, y0, zNk)]
               [v(x1, y1, z0), v(x1, y1, z1), ..., v(x1, y1, zNk)]
               ...
               [v(x1, yNj, z0), v(x1, yNj, z1), ..., v(x1, yNj, zNk)]  ]

            [  ...   ]

            [  [v(xNi, y0, z0), v(xNi, y0, z1), ..., v(xNi, y0, zNk)]
               [v(xNi, y1, z0), v(xNi, y1, z1), ..., v(xNi, y1, zNk)]
               ...
               [v(xNi, yNj, z0), v(xNi, yNj, z1), ..., v(xNi, yNj, zNk)]  ]  ]
   """

   Ni = len(values3D)
   Nj = len(values3D[0])
   Nk = len(values3D[0][0])

   if order == 'xyz':
      values3D = np.transpose(values3D, axes = (2, 1, 0))
      
      '''
      inter_xy_values = np.zeros(Nk)
      
      for k in range(0, Nk):
         inter_y_values = np.zeros(Nj)
         for j in range(0, Nj):
            inter_y_values[j] = inter_1D(x_coords, values_zyx[k][j], pos[0])
         
         inter_xy_values[k] = inter_1D(y_coords, inter_y_values, pos[1])

      inter_xyz_value = inter_1D(z_coords, inter_xy_values, pos[2])
      return inter_xyz_value
      '''

      d1_coords = x_coords
      d2_coords = y_coords
      d3_coords = z_coords

      pos_d1 = pos[0]
      pos_d2 = pos[1]
      pos_d3 = pos[2]

   if order == 'xzy':
      values3D = np.transpose(values3D, axes = (1, 2, 0))
      
      d1_coords = x_coords
      d2_coords = z_coords
      d3_coords = y_coords

      pos_d1 = pos[0]
      pos_d2 = pos[2]
      pos_d3 = pos[1]

   if order == 'yxz':
      values3D = np.transpose(values3D, axes = (2, 0, 1))

      d1_coords = y_coords
      d2_coords = x_coords
      d3_coords = z_coords

      pos_d1 = pos[1]
      pos_d2 = pos[0]
      pos_d3 = pos[2]

   if order == 'yzx':
      values3D = np.transpose(values3D, axes = (0, 2, 1))

      d1_coords = y_coords
      d2_coords = z_coords
      d3_coords = x_coords

      pos_d1 = pos[1]
      pos_d2 = pos[2]
      pos_d3 = pos[0]

   if order == 'zxy':
      values3D = np.transpose(values3D, axes = (1, 0, 2))

      d1_coords = z_coords
      d2_coords = x_coords
      d3_coords = y_coords

      pos_d1 = pos[2]
      pos_d2 = pos[0]
      pos_d3 = pos[1]

   if order == 'zyx':
      # values3D = np.transpose(values3D, axes = (0, 1, 2))

      d1_coords = z_coords
      d2_coords = y_coords
      d3_coords = x_coords

      pos_d1 = pos[2]
      pos_d2 = pos[1]
      pos_d3 = pos[0]
   
   Nd1 = len(values3D[0][0])
   Nd2 = len(values3D[0])
   Nd3 = len(values3D)

   return slicing(d1_coords, d2_coords, d3_coords, Nd1, Nd2, Nd3, values3D, pos_d1, pos_d2, pos_d3)