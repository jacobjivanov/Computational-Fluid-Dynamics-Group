# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
from numpy.fft import fft

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k
# fourier term indexes: f

def inter_1D(coords, values, pos):
   '''
   returns the one dimensional FFT interpolated value at the given position
   
   Parameters: 
      coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array
         Example:
         [x0,  x1,  ..., xNi]
      values : ArrayLike
         a list or numpy.array of values associated with the coord_array
      pos : Float
         A float with the coordinate of the desired interpolated value
         Example: x
   '''
   
   Ni = len(values) #n
   c_max = coords[-1] + coords[1]

   values_fft = fft(values)
   # print(values_fft)
   # print('###########')
   # values_fft = fft(values) / Ni

   value_inter = np.real(values_fft[0]) / Ni
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 / Ni * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 / Ni * np.imag(values_fft[f]) * np.sin(w * pos)
      # print('2 * real({0:.3f}) * cos({1:.3f} * x) - 2 * imag({2:.3f}) * sin({3:.3f} * x)'.format(values_fft[f], w, values_fft[f], w))

   return value_inter

def inter_2D(x_coords, y_coords, values2D, pos):
   '''
   returns the two dimensional FFT interpolated value at the given position
   
   Parameters: 
      x_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [  [x0, x1, ..., xNi],
            [x0, x1, ..., xNi], 
            ...
            [x0, x1, ..., xNi]   ]
      y_coords : ArrayLike
         a list or numpy.array of coordinates associated with the value_array. 
         Example: 
         [  [y0, y0,  ..., y0],
            [y1,  y1,  ..., y1], 
            ...
            [yNj, yNj, ..., yNj]   ]
      values : ArrayLike
         a list or numpy.array of values associated with the coord_array
      pos : List
         A list with the coordinate of the desired interpolated value
         Example: [x, y]
   '''

   Ni = len(values2D[0])

   values1D = np.zeros(Ni)

   for i in range(0, Ni):
      values1D[i] = inter_1D(y_coords.transpose()[i], values2D.transpose()[i], pos[1])

   return inter_1D(x_coords[0], values1D, pos[0])

def inter_3D(x_coords, y_coords, z_coords, values3D, pos):
