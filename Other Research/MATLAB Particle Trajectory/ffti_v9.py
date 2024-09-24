# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
x_dom = [- np.pi, np.pi]
y_dom = [- np.pi, np.pi]
z_dom = [- np.pi, np.pi]

Ni, Nj, Nk = 100, 100, 100

x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], Nj, endpoint = False)
z = np.linspace(z_dom[0], z_dom[1], Nk, endpoint = False)

values1D = np.zeros(shape = (Ni))
values2D = np.zeros(shape = (Ni, Nj))
values3D = np.zeros(shape = (Ni, Nj, Nk))

for i in range(0, Ni):
   for j in range(0, Nj):
      for k in range(0, Nk):
         values3D[i, j, k] = np.sin(x[i]) + np.sin(y[j]) + np.sin(z[k])
      values2D[i, j] = np.sin(x[i]) + np.sin(y[j])
   values1D[i] = np.exp(np.sin(x[i]))

values1D_fft = np.fft.fftn(values1D)
values2D_fft = np.fft.fftn(values2D)
values3D_fft = np.fft.fftn(values3D)

kx = np.array(list(range(0, Ni//2 + 1))) * 2 * np.pi / (x_dom[1] - x_dom[0])
ky = np.array(list(range(0, Nj//2 + 1))) * 2 * np.pi / (y_dom[1] - y_dom[0])
kz = np.array(list(range(0, Nk//2 + 1))) * 2 * np.pi / (z_dom[1] - z_dom[0])

# kx =  np.array(list(range(0, Ni//2 + 1)) + list(range(-Ni//2 + 1, 0)))
# ky =  np.array(list(range(0, Nj//2 + 1)) + list(range(-Nj//2 + 1, 0)))
# kz =  np.array(list(range(0, Nk//2 + 1)) + list(range(-Nk//2 + 1, 0)))

def inter_3D(valuess3D_fft, kx, ky, kz, x, y, z):
   return 0

def inter_1D(values1D_fft, kx, x_pos):
   value_inter = np.real(values1D_fft[0])
   # value_inter = 0
   for k in range(1, len(kx)):
      # print(kx[k])
      value_inter += 2 * np.real(values1D_fft[k]) * np.cos(kx[k] * x_pos)
      value_inter -= 2 * np.imag(values1D_fft[k]) * np.sin(kx[k] * x_pos)
   value_inter /= len(kx) * 2
   return value_inter

a = np.linspace(-5, 5)
b = np.linspace(-5, 5)

b[0] = inter_1D(values1D_fft, kx, a[0])

for i in range(len(a)):
   b[i] = inter_1D(values1D_fft, kx, a[i])
A = np.fft.fftn(values1D)
B = np.fft.fft(values1D)

# print(A)
# print()
# print(B)
import matplotlib.pyplot as plt
plt.plot(x, values1D, '-o')
plt.plot(a, b)
plt.show()