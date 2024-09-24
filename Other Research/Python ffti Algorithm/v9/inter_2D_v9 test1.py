import time
t0 = time.time()

import numpy as np
import ffti_v10 as fi9

def f(x, y):
    return np.sin(x)
    # return np.exp(np.sin(x) + np.sin(y))
    # return np.sin(y) - np.exp(np.sin(2 * y)) - np.sin(15 * x)
    # return np.sin(y) + np.sin(x)

x_dom, Ni = [0, 2 * np.pi], 50
y_dom, Nj = [0, 2 * np.pi], 50

x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], Nj, endpoint = False)

kx = fi9.wavenumbers(Ni, x_dom)
ky = fi9.wavenumbers(Nj, y_dom)

rho = np.empty(shape = (Ni, Nj), dtype = 'complex')
for i in range(0, Ni):
    for j in range(0, Nj):
        rho[i, j] = f(x[i], y[j])
rho_fft = np.fft.fftn(rho)

Ni_inter = 4 * Ni
Nj_inter = 4 * Nj

x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter, endpoint = False)
y_inter = np.linspace(y_dom[0], y_dom[1], Nj_inter, endpoint = False)

rho_approx = np.zeros(shape = (Ni_inter, Nj_inter), dtype = 'complex')
rho_error = np.zeros(shape = (Ni_inter, Nj_inter), dtype = 'complex')

for i in range(0, Ni_inter):
    for j in range(0, Nj_inter):
        # print([x_inter[i], y_inter[j]])
        rho_approx[i, j] = fi9.inter_2D(kx, ky, rho_fft, [x_inter[i], y_inter[j]])
        # print(rho_approx[i, j])
        rho_error[i, j] = rho_approx[i][j] - f(x_inter[i], y_inter[j])

t1 = time.time()

print(t1 - t0)

import matplotlib.pyplot as plt
fig1 = plt.figure()
# fig1.tight_layout()
ax1 = fig1.add_subplot(projection = '3d')

# Discrete Input Scatter Plot
# ax1.scatter(np.meshgrid(x, y, indexing = "ij")[0], np.meshgrid(x, y, indexing = "ij")[1], np.real(rho), color = "grey", alpha = 1, label = "discrete input")

# Fourier Interpretation Surface Plot
ax1.plot_surface(np.meshgrid(x_inter, y_inter, indexing = "ij")[0], np.meshgrid(x_inter, y_inter, indexing = "ij")[1], np.real(rho_error), alpha = 0.5, cmap = 'cool', label = "Continuous Interpolation")
# , cmap = "cool"
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
ax1.set_zlabel(r"$\Re \left[\rho \right]$")
ax1.set_title("Real Component of Discrete and Interpolated Function")

ax1.view_init(azim = 225, elev = 45)
# ax1.legend()
plt.show(block = False)

fig2 = plt.figure()
# fig2.tight_layout()
ax2 = fig2.add_subplot(projection = '3d')

# Discrete Input Scatter Plot
# ax2.scatter(np.meshgrid(x, y, indexing = "ij")[0], np.meshgrid(x, y, indexing = "ij")[1], np.imag(rho), color = "grey", alpha = 1, label = "discrete input")

# Fourier Interpretation Surface Plot
ax2.plot_surface(np.meshgrid(x_inter, y_inter, indexing = "ij")[0], np.meshgrid(x_inter, y_inter, indexing = "ij")[1], np.imag(rho_error), alpha = 0.5, cmap = 'cool', label = "Continuous Interpolation")
# cmap = "cool",

ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax2.set_zlabel(r"$\Im \left[ \rho \right]$")
ax2.set_title("Imaginary Component of Discrete and Interpolated Function")

ax2.view_init(azim = 225, elev = 45)
# ax2.legend()
plt.show()