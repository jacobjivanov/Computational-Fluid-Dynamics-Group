# The following script was written by Jacob Ivanov, Undergraduate Researcher for the Computational Fluid Dynamics Group at the University of Connecticut, under Dr. Georgios Matheou. 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 3, figsize = (6.5, 3))
for i in range(0, 3):
    M = [32, 256, 2048][i]
    x = np.linspace(0, 2*np.pi, M, endpoint = False)
    y = np.linspace(0, 2*np.pi, M, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    k = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

    psi = np.exp(np.cos(x_grid)) - np.sin(y_grid)
    omega = np.exp(np.cos(x_grid)) * (np.cos(x_grid) - np.sin(x_grid)**2) - np.sin(y_grid)

    Omega = np.fft.fft2(omega)
    k2 = np.zeros(shape = (M, M))
    for p in range(0, M):
        for q in range(0, M):
            k2[p, q] = (k[p]**2 + k[q]**2) if [p, q] != [0, 0] else np.inf

    Psi = Omega / k2
    psi_num = np.real(np.fft.ifft2(Psi)) + np.average(psi)
    
    # figure configurations below
    plot = ax[i].pcolormesh(x, y, psi_num - psi, cmap = 'coolwarm')
    ax[i].set_aspect('equal')

    ax[i].set_xlim(0, 2 * np.pi)
    ax[i].set_ylim(0, 2 * np.pi)

    ax[i].set_xlabel(r"$x$")
    ax[i].set_title("Grid: {0} ✕ {0}".format(M))
    if i == 0:
        ax[i].set_ylabel(r"$y$")

    ax[i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))

    ax[i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax[i].yaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax[i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))

    divider = make_axes_locatable(plot.axes)
    cax = divider.append_axes("bottom", size = "5%", pad = 0.5)
    
    fig.colorbar(plot, cax = cax, orientation = 'horizontal')

fig.suptitle("Error in " + r"$\psi_\mathrm{num}$" + " For Varying Grid Size")

plt.tight_layout()
plt.savefig("spectral_poisson.png", dpi = 200)