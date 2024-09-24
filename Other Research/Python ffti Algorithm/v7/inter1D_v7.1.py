import numpy as np

def inter_1D(coords, values, pos):
   return reconstruction(np.fft.fft(values[:-1]), len(values) - 1, coords[-1], pos)

def reconstruction(values_fft, Ni, c_max, pos):
   value_inter = np.real(values_fft[0])
   for f in range(1, Ni // 2 + 1):
      w = f * 2 * np.pi / (2 * c_max)
      value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
      value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)
   
   value_inter /= Ni
   return value_inter 

def inter_1D_v7_1(values_fft, Ni, x_dom, pos):
    value_inter = np.real(values_fft[0])
    for f in range(1, Ni // 2 + 1):
        w = f * 2 * np.pi / (x_dom[1] - x_dom[0])
        value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
        value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)

    value_inter /= Ni
    return value_inter

x_dom = [- np.pi, np.pi]
Ni = 100
x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
x_v7 = np.linspace(x_dom[0], x_dom[1], Ni + 1)

values1D = np.zeros(shape = (Ni))
values1D_v7 = np.zeros(shape = (Ni + 1))

for i in range(0, Ni):
    values1D[i] = np.exp(np.sin(x[i]))
    values1D[i] = np.exp(np.sin(x_v7[i]))

values1D_fft = np.fft.fftn(values1D)

a = np.linspace(-5, 5)
b = np.linspace(-5, 5)
b_v7 = np.linspace(-5, 5)

for i in range(len(a)):
    b[i] = inter_1D_v7_1(values1D_fft, Ni, x_dom, a[i])
    b_v7[i] = inter_1D(x_v7, values1D_v7, a[i])

import matplotlib.pyplot as plt
plt.plot(x, values1D, '-o')
plt.plot(a, b)
plt.plot(a, b_v7)
plt.show()