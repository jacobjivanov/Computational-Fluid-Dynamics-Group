import numpy as np
import matplotlib.pyplot as plt

x_dom = [- np.pi, np.pi]

Ni = 10

x = np.linspace(x_dom[0], x_dom[1], Ni + 1)[:-1]
y = np.empty(Ni)

for i in range(Ni):
    y[i] = np.sin(x[i])

Y = np.fft.fft(y)
print(Y)