# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
import matplotlib.pyplot as plt

def La_Norm(e, dx, dy, a):
    N = len(e)
    La = 0
    for i in range(0, N):
        La += np.abs(e[i]) ** a
    La = (dx * dy * La) ** (1/a)
    return La

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
L_x, L_y = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0]
M = 256
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/M

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], M, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
k = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

psi = np.exp(np.cos(x_grid)) - np.sin(y_grid)
omega = np.exp(np.cos(x_grid)) * (np.cos(x_grid) - np.sin(x_grid)**2) - np.sin(y_grid)

Omega = np.fft.fft2(omega)
psi_hat = np.zeros(shape = (M, M), dtype = 'complex')
k2 = np.zeros(shape = (M, M))
for p in range(0, M):
    for q in range(0, M):
        k2[p, q] = (-k[p]**2 - k[q]**2) if [p, q] != [0, 0] else np.inf

Psi = Omega / k2
psi_num = np.real(np.fft.ifft2(Psi))


fig, ax = plt.subplots()
fig.suptitle(r"$\psi_{\mathrm{recovered}} + \psi_{\mathrm{defined}}$")
omega_con = ax.contourf(x, y, psi + psi_num, cmap = 'coolwarm')
ax.set_aspect('equal')
fig.colorbar(omega_con)
plt.show()