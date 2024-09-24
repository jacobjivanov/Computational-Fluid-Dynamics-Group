# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
from numpy.fft import fft

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k
# fourier term indexes: f

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
   
   Ni = len(values)
   c_max = coords[-1] + coords[1]

   values_fft = fft(values)

   value_inter = np.real(values_fft[0]) / Ni
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 / Ni * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 / Ni * np.imag(values_fft[f]) * np.sin(w * pos)

   return value_inter

def inter_2D(x_coords, y_coords, values2D, pos, order = 'xy'):
   """
   returns the two dimensional FFT interpolated value at the given position
   
   Parameters: 
      x_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [  [x0, x0, ..., x0],
            [x1, x1, ..., x1], 
            ...
            [xNi, xNi, ..., xNi]   ]
      y_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [  [y0, y1, ..., yNj],
            [y0, y1, ..., yNj], 
            ...
            [y0, y1, ..., yNj]   ]
      values : ArrayLike
         a list or numpy.array of values associated with the coord_array
         Example:
         [  [v(x0, y0), v(x0, y1), ..., v(x0, yNj)],
            [v(x1, y0), v(x1, y1), ..., v(x1, yNj)],
            ...
            [v(xNi, y0), v(xNi, y1), ..., v(xNi, yNj)]   ]
      pos : List
         A list with the coordinate of the desired interpolated value
         Example: [x, y]
   """

   Ni = len(values2D)
   Nj = len(values2D[0])

   if order == 'xy': # interpolate at x first, then at y
      values1D = np.zeros(Nj)

      for j in range(0, Nj):
         values1D[j] = inter_1D(x_coords.transpose()[0], values2D.transpose()[j], pos[0])

      return inter_1D(y_coords[0], values1D, pos[1])
   
   if order == 'yx': # interpolate at y first, then at x
      return 0


def inter_3D(x_coords, y_coords, z_coords, values3D, pos, order = 'xyz'):
   Ni = len(values3D)
   Nj = len(values3D[0])
   Nk = len(values3D[0][0])

   values1D = np.zeros(Nk)

   for k in range(0, Nk):
      print("x_coords: ")
      print(x_coords)
      values1D[k] = inter_2D(x_coords, y_coords, values3D[:, :, k], pos[0:1])
   
   return inter_1D(z_coords, values1D, pos[2])