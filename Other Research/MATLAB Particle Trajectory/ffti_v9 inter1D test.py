# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np

x_dom = [- np.pi, np.pi]
Ni = 100
x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)

values1D = np.zeros(shape = (Ni))
for i in range(0, Ni):
   values1D[i] = np.exp(np.sin(x[i]))

values1D_fft = np.fft.fftn(values1D)

# kx = np.array(list(range(0, Ni//2 + 1))) * 2 * np.pi / (x_dom[1] - x_dom[0])
kx =  np.array(list(range(0, Ni//2 + 1)) + list(range(-Ni//2 + 1, 0)))

def inter_1D(values1D_fft, kx, x_pos):
   value_inter = 0
   for k in range(0, len(kx)):
      value_inter += 2 * np.real(values1D_fft[k]) * np.cos(kx[k] * x_pos)
      value_inter -= 2 * np.imag(values1D_fft[k]) * np.sin(kx[k] * x_pos)
   value_inter /= len(kx) * 2
   return value_inter

a = np.linspace(-5, 5)
b = np.linspace(-5, 5)

for i in range(len(a)):
   b[i] = inter_1D(values1D_fft, kx, a[i])

import matplotlib.pyplot as plt
plt.plot(x, values1D, '-o')
plt.plot(a, b)
plt.show()