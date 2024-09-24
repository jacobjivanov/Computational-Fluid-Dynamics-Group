import numpy as np
from numpy.fft import fft
import ffti_v6 as fi

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
   
   Ni = len(values) - 1
   c_max = coords[-1]
   print(c_max)

   values_fft = fft(values[:-1])

   value_inter = np.real(values_fft[0]) / Ni
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 / Ni * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 / Ni * np.imag(values_fft[f]) * np.sin(w * pos)

   return value_inter

time = [0.0,0.5714285714285714,1.1428571428571428,1.7142857142857142,2.2857142857142856,2.8571428571428568,3.4285714285714284,4.0,4.571428571428571,5.142857142857142,5.7142857142857135,6.285714285714286,6.857142857142857,7.428571428571428,8.0,];
   
signal = [0.0,0.5408342133588315,0.9098229129411239,0.9897230488598214,0.7551470262316581,0.2806293995143573,-0.2830558540822556,-0.7568024953079283,-0.9900815210958355,-0.9087704868046733,-0.538705288386157,0.002528975838921635,0.5429596793024328,0.910869520096767,0.9893582466233818,];
   

a = fi.inter_1D(time, signal, 5.76112)
print(a)
