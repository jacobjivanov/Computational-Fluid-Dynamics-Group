import numpy as np
import ffti_v10 as fi10

def f(x, y):
    # return np.exp(np.sin(x) - np.cos(z * 3 * y))
    return np.exp(np.sin(x) + np.sin(y))

def linf_error(Ni):
    x_dom, Ni = [0, 2 * np.pi], Ni
    y_dom, Nj = [0, 2 * np.pi], Ni

    x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
    y = np.linspace(y_dom[0], y_dom[1], Nj, endpoint = False)

    kx = fi10.wavenumbers(Ni, x_dom)
    ky = fi10.wavenumbers(Nj, y_dom)

    rho = np.empty(shape = (Ni, Nj))
    for i in range(0, Ni):
        for j in range(0, Nj):
            rho[i, j] = f(x[i], y[j])
    rho_fft = np.fft.fftn(rho)


    Ni_inter = 100
    Nj_inter = 100

    x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter, endpoint = False)
    y_inter = np.linspace(x_dom[0], x_dom[1], Nj_inter, endpoint = False)

    s = 0
    for ii in range(0, Ni_inter):
        for ji in range(0, Nj_inter):
            a = abs(np.real(fi10.inter_2D(kx, ky, rho_fft, [x_inter[ii], y_inter[ji]])) - f(x_inter[ii], y_inter[ji]))
            s = a if a > s else s
            # s += a ** 2
            # print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (ii + ((ji + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
    # print("Interpolating. \tProgress: 100.000% Complete")


    # s *= (x_inter[1] * y_inter[1])/(Ni_inter * Nj_inter)
    # s = np.sqrt(s)

    return s

error = np.zeros(32)
n = np.arange(2, 66, 2)
for i in range(32):
    error[i] = linf_error(n[i])
    print(n[i], error[i])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

ax.scatter(n, error)
ax.axhline(2.22044604925e-16, label = "64 Bit Precision", linestyle = 'dashed', color = 'gray')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Error Convergence of inter_2D()")
ax.set_xlabel("Ni")
ax.set_ylabel("L∞-Norm of Signed Interpolation Error")
ax.set_ylim(1e-16, 1e2)
ax.legend(loc = "lower left")

plt.show()