# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
from numpy.fft import fft
from numba import njit

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k
# fourier term indexes: f

def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
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

def inter_2D(x_coords, y_coords, values2D, pos):
   Ni, Nj = values2D.shape

   inter_y_values = np.zeros(Ni)
   for i in range(0, Ni):
      inter_y_values[i] = inter_1D(y_coords, values2D[i], pos[1])

   inter_yx_value = inter_1D(x_coords, inter_y_values, pos[0])
   return inter_yx_value

def inter_3D(x_coords, y_coords, z_coords, values3D, pos):
   """
   returns the three dimensional FFT interpolated value at the given position
   
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

   Ni, Nj, Nk = values3D.shape

   inter_zy_values = np.zeros(Ni)

   for i in range(0, Ni):
      inter_y_values = np.zeros(Nj)
      for j in range(0, Nj):
         inter_y_values[j] = inter_1D(z_coords, values3D[i][j], pos[2])

      inter_zy_values[i] = inter_1D(y_coords, inter_y_values, pos[1])

   inter_zyx_value = inter_1D(x_coords, inter_zy_values, pos[0])
   return inter_zyx_value