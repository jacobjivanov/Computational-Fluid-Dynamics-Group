import numpy as np

def inter_1D2(coords, values, pos):
    values_fft = np.fft.fftn(values)
    print(values_fft)
    Ni = len(values)
    c_max = coords[-1] + coords[1]

    value_inter = np.real(values_fft[0])
    for f in range(1, Ni // 2 + 1):
        w = f * 2 * np.pi / c_max
        value_inter += 2 * np.real(values_fft[f]) * np.cos(w * pos)
        value_inter -= 2 * np.imag(values_fft[f]) * np.sin(w * pos)

    value_inter /= Ni
    return value_inter

Ni = 12
x_dom = [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)

y = np.zeros(Ni)
for i in range(Ni):
   # y[i] = np.sin(x[i]) - 3 * np.cos(3 * x[i])
   # y[i] = np.exp(np.sin(x[i]) - 5 * np.sin(x[i] * 3) * np.cos(x[i]))
   y[i] = np.exp(np.sin(2 * x[i]))
y_fft = np.fft.fftn(y)

Ni_inter = 1000
x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter)
y_inter = np.zeros(Ni_inter)
for i in range(Ni_inter):
   y_inter[i] = inter_1D2(x, y, x_inter[i])

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x_inter, y_inter, color = 'red')

plt.show()
