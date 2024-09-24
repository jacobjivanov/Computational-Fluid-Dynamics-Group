import numpy as np
import scipy
import numba as nb

def La_norm(e, dx, dy, a):
    N = len(e)
    La = 0
    for i in range(0, N):
        La += np.abs(e[i]) ** a
    La = (dx * dy * La) ** (1/a)
    return La

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

@nb.njit()
def inter_2D(kp, kq, U, pos):
    M, N = U.shape

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, N):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * pos[0])

    u_inter /= M
    return np.real(u_inter)

import numpy as np
from numba import njit

@nb.njit
def interpolate(coords, values, pos):
    Ni = len(coords)
    pos = pos % (coords[Ni - 1] - coords[0]) + coords[0]
    i = 0
    while coords[i] < pos:
        i += 1
        if i == Ni:
            i = 0

    value_inter = values[i - 1] + (pos - coords[i - 1]) * (values[i] - values[i - 1]) / (coords[i] - coords[i - 1])
    return value_inter

@nb.njit
def inter_2D_linear(x_coords, y_coords, values2D, pos):
    Ni = len(values2D)
    Nj = len(values2D[0])

    values1D = np.zeros(Nj)
    for j in range(Nj):
        values1D[j] = interpolate(x_coords, values2D.transpose()[j], pos[0])
    value_inter = interpolate(y_coords, values1D, pos[1])

    return value_inter

import numpy as np
from numba import njit

@nb.njit
def inter_2D_nearest(x_coords, y_coords, values2D, pos):
    Ni = len(values2D)
    Nj = len(values2D[0])

    values1D = np.zeros(Nj)
    pos_x, pos_y = pos

    pos_x = pos_x % (x_coords[Ni - 1] - x_coords[0]) + x_coords[0]

    i = 0
    while x_coords[i] < pos_x:
        i += 1
        if i == Ni:
            i = 0

    for j in range(Nj):
        values1D[j] = values2D[i - 1, j] if abs(pos_y - y_coords[j]) < abs(pos_y - y_coords[j + 1]) else values2D[i, j]

    return values1D[0] if abs(pos_x - x_coords[i - 1]) < abs(pos_x - x_coords[i]) else values1D[1]

dim = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 48, 56, 64, 128, 256, 512, 1024])
l2_spectral = np.empty(len(dim))
l2_linear   = np.empty(len(dim))
l2_nearest  = np.empty(len(dim))

for d in range(0, len(dim)):
    M, N = dim[d], dim[d]
    x = np.linspace(0, 2 * np.pi, M, endpoint = False)
    y = np.linspace(0, 2 * np.pi, N, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')

    kp, kq = wavearrays(M, N)

    u = np.exp(np.sin(x_grid) * np.cos(y_grid))
    U = np.fft.fft2(u)

    test = dim[d]
    e_spectral, e_linear, e_nearest = np.empty(test), np.empty(test), np.empty(test)
    for point in range(0, test):
        x_int, y_int = np.random.rand(2) * 2 * np.pi
        u_spectral = inter_2D(kp, kq, U, pos = [x_int, y_int])
        u_linear = inter_2D_linear(x, y, u, pos = [x_int, y_int])
        u_nearest = inter_2D_nearest(x, y, u, pos = [x_int, y_int])

        u_ana = np.exp(np.sin(x_int) * np.cos(y_int))
        e_spectral[point] = np.abs(u_spectral - u_ana)
        e_linear[point] = np.abs(u_linear - u_ana)
        e_nearest[point] = np.abs(u_nearest - u_ana)
        # print("{:.5e}, {:.5e}".format(e_spectral[point], e_linear[point]))

    l2_spectral[d] = La_norm(e_spectral, x[1], y[1], a = 2)
    l2_linear[d] = La_norm(e_linear, x[1], y[1], a = 2)
    l2_nearest[d] = La_norm(e_nearest, x[1], y[1], a = 2)
    print("{0}, {1:.5e}, {2:.5e}, {3:.5e},".format( M, l2_spectral[d], l2_linear[d], l2_nearest[d]))

import matplotlib.pyplot as plt
"""
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
ax.plot_surface(x_grid, y_grid, u, cmap = 'coolwarm', alpha = 0.5)
ax.scatter(x_int, y_int, u_int, color = 'red')
plt.show()
"""

fig, ax = plt.subplots(1, 1, figsize = (6, 3))
ax.loglog(dim, l2_spectral, ':o', color = 'blue', label = 'spectral')
ax.loglog(dim, l2_linear, ':o', color = 'orange', label = 'linear')
ax.loglog(dim, l2_nearest, ':o', color = 'red', label = 'nearest neighbor')
ax.legend()
ax.set_xlabel("Square Grid Dimensions, $d$")
ax.set_ylabel("$\ell_2(u_{\mathrm{int}} - u_{\mathrm{ana}})$")
# plt.show()
plt.savefig("Comparison of Interpolation Convergence by Method.png", dpi = 200, bbox_inches = 'tight')