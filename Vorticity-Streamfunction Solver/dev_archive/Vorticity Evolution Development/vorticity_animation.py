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

M, N = 256, 256 # computational grid size
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

# defining various wavenumber constants
def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

k_p, k_q = wavearrays(M, N)
with np.errstate(divide = 'ignore', invalid = 'ignore'): # [p, q] = [0, 0] singularity
    k_U = +1j*k_q / (k_p**2 + k_q**2)
    k_V = -1j*k_p / (k_p**2 + k_q**2)
k_U[0, 0], k_V[0, 0] = 0, 0

@nb.njit()
def velocities(Omega):
    U, V = k_U * Omega, k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))
    uv_mag = (u**2 + v**2) ** (1/2)

    return uv_mag, u, v 

Omega = np.load("Vorticity Evolution Development/temp/v6 256 unforced 0.02/Omega n = 0.npy")
t = np.load("Vorticity Evolution Development/temp/v6 256 unforced 0.02/t n = 0.npy")

uv_mag, u, v = velocities(Omega)
eo = M // 32
fig, ax = plt.subplots(1, 1, constrained_layout = True)
uv_plot = ax.pcolormesh(x_grid, y_grid, uv_mag, cmap = 'coolwarm')
uv_vecs = ax.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo],
                    u[::eo, ::eo], v[::eo, ::eo], pivot = 'mid', alpha = 0.5, color = 'grey')

ax.set_aspect('equal')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.xaxis.set_major_locator(MultipleLocator(base = np.pi))
ax.yaxis.set_major_locator(MultipleLocator(base = np.pi))

divider = make_axes_locatable(uv_plot.axes)
cax = divider.append_axes("right", size = "5%", pad = 0.2)
fig.colorbar(uv_plot, cax = cax, orientation = 'vertical')
cax.set_ylabel(r"$\mathrm{mag}\left[ \vec{u} \right]$")

ax.set_title(r"$\vec{u}(t = $" + "0)")

def update(n):
    print(n)
    Omega = np.load("Vorticity Evolution Development/temp/v6 256 unforced 0.02/Omega n = {0}.npy".format(n))
    t = np.load("Vorticity Evolution Development/temp/v6 256 unforced 0.02/t n = {0}.npy".format(n))

    uv_mag, u, v = velocities(Omega)

    uv_plot.set_array(uv_mag)
    uv_vecs.set_UVC(u[::eo, ::eo], v[::eo, ::eo])
    ax.set_title(r"$\vec{u}(t = $" + "{0:.5f})".format(t))
    uv_plot.set_norm(Normalize(vmin = uv_mag.min(), vmax = uv_mag.max()))

ANI = FuncAnimation(fig, update, init_func = lambda : None, frames = range(0, 15400, 50))
ANI.save("Vorticity Evolution Development/anim/v6 256 unforced 0.02.mp4", dpi = 200, fps = 60, writer = 'ffmpeg')