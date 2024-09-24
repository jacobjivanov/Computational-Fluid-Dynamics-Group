import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

M = 256 # computational grid size
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, M, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]
nu = 5e-3
beta = 1
eo = M // 32
N = 100

data = pd.read_csv("inertial_particles/particle_positions.csv")

x_pos = np.empty(shape = (len(data), N))
y_pos = np.empty(shape = (len(data), N))

for n in range(0, N):
    x_pos[:, n] = data["x{0}".format(n)]
    y_pos[:, n] = data["y{0}".format(n)]
t_data = np.array(data["t"])

def velocities(t):
    u = np.exp(-nu*t) * np.cos(beta*x_grid) * np.sin(beta*y_grid)
    v = - np.exp(-nu*t) * np.sin(beta*x_grid) * np.cos(beta*y_grid)

    uv_mag = np.sqrt(u**2 + v**2)

    return u, v, uv_mag

u, v, uv_mag = velocities(0)

fig, ax = plt.subplots(1, 1, constrained_layout = False)
uv_plot = ax.pcolormesh(x_grid, y_grid, uv_mag, cmap = 'coolwarm')
uv_vecs = ax.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo], u[::eo, ::eo], v[::eo, ::eo])
part_scat = ax.scatter(x_pos[0, :], y_pos[0, :])
ax.set_title(r"$\vec{u}$")

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

def update(n):
    t = t_data[n]
    print("{0:.5f}".format(t), end = '\r')
    u, v, uv_mag = velocities(t)

    part_scat.set_offsets(np.stack((x_pos[n, :], y_pos[n, :]), axis = 1))
    uv_plot.set_array(uv_mag)
    uv_vecs.set_UVC(u[::eo, ::eo], v[::eo, ::eo])
    uv_plot.set_norm(Normalize(vmin = uv_mag.min(), vmax = uv_mag.max()))

ANI = FuncAnimation(fig, update, init_func = lambda : None, frames = range(0, len(data), 1))
ANI.save("Inertial Particles.mp4", fps = 180, dpi = 200, writer = 'ffmpeg')