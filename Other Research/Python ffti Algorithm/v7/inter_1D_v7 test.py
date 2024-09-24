import numpy as np

def inter_1D(coords, values, pos):
   return reconstruction(np.fft.fft(values[:-1]), len(values) - 1, coords[-1], pos)

def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / c_max
      value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
   value_inter /= Ni
   return value_inter 

Ni = 20
x_dom = [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], Ni)

y = np.zeros(Ni)
for i in range(Ni):
   y[i] = np.exp(np.sin(x[i]) - 5 * np.cos(x[i]))

Ni_inter = 100
x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter)
y_inter = np.zeros(Ni_inter)
for i in range(Ni_inter):
   y_inter[i] = inter_1D(x, y, x_inter[i])

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x_inter, y_inter, color = 'red')

plt.show()