import numpy as np
import ffti_v10 as fi9

"""
def inter_1D(kx, values_fft, pos):
    # print(len(values_fft))
    # Ni = len(values_fft)
    value_inter = 0

    for f in range(0, len(values_fft)):
        value_inter += np.real(values_fft[f]) * np.cos(kx[f] * pos)
        value_inter -= np.imag(values_fft[f]) * np.sin(kx[f] * pos)

    value_inter /= len(values_fft)
    return value_inter
"""
    
Ni = 20
# x_dom = [0, 10]
x_dom = [0, 2 * np.pi]
# x_dom = [- np.pi, np.pi]
# x_dom = [- np.pi * 4, -2 * np.pi]
# x_dom = [np.pi * 2, 4 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)

y = np.zeros(Ni)
for i in range(Ni):
    y[i] = np.exp(np.sin(x[i]) - 5 * np.sin(x[i] * 3) * np.cos(x[i]))
    # y[i] = np.exp(np.sin(x[i]) - np.cos(3 * x[i]))
    # y[i] = np.exp(np.sin(2 * x[i]))
    # y[i] = 1 if x[i] > np.pi else 0
y_fft = np.fft.fftn(y)

kx = np.array(list(range(0, Ni//2 + 1)) + list(range(-Ni//2 + 1, 0))) * 2 * np.pi / (x_dom[1] - x_dom[0])
print(kx)
Ni_inter = 1000

x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter)
# print(x)
# print(x_inter)
y_inter = np.zeros(Ni_inter)
for i in range(Ni_inter):
    y_inter[i] = fi9.inter_1D(kx, y_fft, x_inter[i])

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x_inter, y_inter, color = 'red')

plt.show()

# print(fi9.inter_1D(kx, y_fft, 1.3))