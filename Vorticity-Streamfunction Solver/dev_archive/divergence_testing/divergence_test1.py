import numpy as np
from numpy import pi, real
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def recover_dil(U, V):
    Dil = np.empty(shape = (M, N), dtype = 'complex') # frequency dilatation
    
    for p in range(0, M):
        for q in range(0, N):
            Dil[p, q] = 1j * (kx[p] + U[p, q] + ky[q] + V[p, q])

    return np.real(ifft2(Dil))

# computational grid parameters
M, N = 32, 32
x_dom, y_dom = [0, 2 * pi], [0, 2 * pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
dx, dy = x[1] - x[0], y[1] - y[0]
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

u = np.empty(shape = (M, N)) # physcial velocity x-component
v = np.empty(shape = (M, N)) # physical velocity y-component

for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(x[i]) * np.sin(y[j])
        v[i, j] = - np.sin(x[i]) * np.cos(y[j])

U, V = fft2(u), fft2(v) # frequency velocity x-component, y-component
"""
Dil = np.empty(shape = (M, N), dtype = "complex") # frequency dilatation
for p in range(0, M):
    for q in range(0, N):
        Dil[p, q] =  1j * (kx[p] * U[p, q] + ky[q] * V[p, q])
dil = np.real(ifft2(Dil))
"""

dil = recover_dil(U, V)

uv = np.empty(shape = (M, N))
for i in range(0, M):
    for j in range(0, N):
        uv[i, j] = (u[i, j] ** 2 + v[i, j] ** 2) ** (1/2)

# FIGURE SETTINGS BELOW

fig, ax = plt.subplots(1, 2, constrained_layout = True)

eo = 4
ax[0].set_title(r"$\vec{u}$")
vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
ax[0].quiver(x[: : eo], y[: : eo], u[: : eo, : : eo], v[: : eo, : : eo])
ax[0].set_aspect('equal')

ax[1].set_title(r"$\nabla \cdot \vec{u}$")
ax[1].set_aspect('equal')
dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

for plt_i in [0, 1]:
    ax[plt_i].set_xlim(0, 2 * pi)
    ax[plt_i].set_ylim(0, 2 * pi)

    ax[plt_i].set_xlabel(r"$x$")
    ax[plt_i].set_ylabel(r"$y$")

    ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = pi))
    ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = pi))

for plot in [vel_plot, dil_plot]:
    divider = make_axes_locatable(plot.axes)
    cax = divider.append_axes("bottom", size = "5%", pad = 0.3)
    fig.colorbar(plot, cax = cax, orientation = 'horizontal')

plt.show()