import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numba as nb
from STKE_helpers import wavearrays

M, N = 256, 256 # computational grid size
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

k_p, k_q, k_U, k_V = wavearrays(M, N)

@nb.njit()
def velocities(Omega):
    U, V = k_U * Omega, k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))
    uv_mag = (u**2 + v**2) ** (1/2)

    return uv_mag, u, v

current_file = np.load("STKE Vorticity Evolution/temp/STKE v1.1 {0}/Omega+Theta_p+t n = 0.npz".format(M))
Omega = current_file['Omega']
theta_p = real(ifft2(current_file['Theta_p']))
t = current_file['t']

uv_mag, u, v = velocities(Omega)
eo = M // 32
fig, ax = plt.subplots(1, 2)
uv_plot = ax[0].pcolormesh(x_grid, y_grid, uv_mag, cmap = 'coolwarm')
uv_vecs = ax[0].quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo],
                    u[::eo, ::eo], v[::eo, ::eo], pivot = 'mid', alpha = 0.5, color = 'grey')

theta_p_plot = ax[1].pcolormesh(x_grid, y_grid, theta_p, cmap = 'viridis')

ax[0].set_ylabel(r"$y$")

for p in range(0, 2):
    ax[p].set_aspect('equal')
    ax[p].set_xlim(0, 2 * np.pi)
    ax[p].set_ylim(0, 2 * np.pi)
    ax[p].set_xlabel(r"$x$")
    ax[p].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax[p].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax[p].xaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax[p].yaxis.set_major_locator(MultipleLocator(base = np.pi))

divider = make_axes_locatable(uv_plot.axes)
cax = divider.append_axes("bottom", size = "5%", pad = 0.3)
fig.colorbar(uv_plot, cax = cax, orientation = 'horizontal')
cax.set_xlabel(r"$\mathrm{mag}\left[ \vec{u} \right]$")
ax[0].set_title(r"$\vec{u}$")

divider = make_axes_locatable(theta_p_plot.axes)
cax = divider.append_axes("bottom", size = "5%", pad = 0.3)
fig.colorbar(theta_p_plot, cax = cax, orientation = 'horizontal')
ax[1].set_title(r"$\theta'$")

fig.suptitle("t = {0:.5f}".format(t), y = 0.9)

def update(n):
    print(n)
    
    current_file = np.load("STKE Vorticity Evolution/temp/STKE v1.1 {0}/Omega+Theta_p+t n = {1}.npz".format(M, n))
    Omega = current_file['Omega']
    theta_p = real(ifft2(current_file['Theta_p']))
    t = current_file['t']

    uv_mag, u, v = velocities(Omega)

    uv_plot.set_array(uv_mag)
    uv_vecs.set_UVC(u[::eo, ::eo], v[::eo, ::eo])
    uv_plot.set_norm(Normalize(vmin = uv_mag.min(), vmax = uv_mag.max()))

    theta_p_plot.set_array(theta_p)
    theta_p_plot.set_norm(Normalize(vmin = theta_p.min(), vmax = theta_p.max()))

    fig.suptitle("t = {0:.5f}".format(t), y = 0.9)

ANI = FuncAnimation(fig, update, init_func = lambda : None, frames = range(0, 36780, 20))
ANI.save("STKE Vorticity Evolution/anim/STKE v1.1 {0}.mp4".format(M), dpi = 200, fps = 60, writer = 'ffmpeg')