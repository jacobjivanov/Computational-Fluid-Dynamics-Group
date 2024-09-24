import numpy as np

# x-axis indexes: i
# y-axis indexes: j
# z-axis indexes: k

def inter_1D(coords, values, pos):
   """
   returns the one dimensional linear interpolated value at the given position
   
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
   """

   if pos in coords:
      return values[list(coords).index(pos)]

   i = 0
   while coords[i] < pos:
      i += 1
   
   slope = (values[i] - values[i - 1]) / (coords[i] - coords[i - 1])
   return values[i - 1] + slope * (pos - coords[i - 1])

def inter_2D(x_coords, y_coords, values2D, pos):
   """
   returns the two dimensional linear interpolated value at the given position
   
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
   """

   Ni = len(values2D[0])

   values1D = np.zeros(Ni)
   coords1D = x_coords[0]

   for i in range(0, Ni):
      values1D[i] = inter_1D(y_coords.transpose()[i], values2D.transpose()[i], pos[1])

   return inter_1D(coords1D, values1D, pos[0])