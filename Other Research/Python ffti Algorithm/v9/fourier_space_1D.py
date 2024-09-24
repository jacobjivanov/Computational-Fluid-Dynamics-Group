import numpy as np
import ffti_v9 as fi9

def f(x):
    return 4 + 3 * np.sin(x) + np.cos(15 * x)

x_dom, Ni = [0, 2 * np.pi], 50
x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
kx = fi9.wavenumbers(Ni, x_dom)

rho = np.empty(shape = (Ni))
for i in range(0, Ni):
    rho[i] = f(x[i])
rho_fft = np.fft.fftn(rho)

import matplotlib.pyplot as plt


# plt.plot(x, rho, '-o')
# plt.show()

plt.plot(kx[0 : Ni//2 + 1], np.log10(rho_fft[0 : Ni//2 + 1]), '-o')
plt.show()