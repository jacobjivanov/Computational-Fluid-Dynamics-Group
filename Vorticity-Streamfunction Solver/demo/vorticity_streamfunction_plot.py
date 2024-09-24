import numpy as np
from numpy.fft import ifft2, fft2
from numpy import real, imag

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


M = 256
N = 50
tau = 1
forced = False
nu = 9e-4
savepath = "{0} {1} {2} {3}".format(M, N, tau, forced)

def init_grid(M):
    x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, M, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    dx, dy = x[1], y[1]

    return x, y, x_grid, y_grid, dx, dy

def init_wavearrays(M):
    kp = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1)).T

    kq = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1))
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        kU0 = +1j*kq / (kp**2 + kq**2)
        kU1 = -1j*kp / (kp**2 + kq**2)
        kXi = kp**2 + kq**2

    # forced turbulence viscosity modification
    if forced == True:
        for p in range(0, M):
            for q in range(0, M):
                if 5**2 <= kXi[p, q] and kXi[p, q] <= 6**2:
                    kXi[p, q] *= -1
                if kXi[p, q] <= 2**2:
                    kXi[p, q] *= 8
    
    kU0[0, 0], kU1[0, 0] = 0, 0
    kU = np.array([kU0, kU1])
    kXi = nu * kXi

    return kp, kq, kU0, kU1, kU, kXi

eo = M // 32
x_grid, y_grid = init_grid(M)[2:4]
kU0, kU1 = init_wavearrays(M)[2:4]

def calc_u(Omega):
    u = np.array([
        real(ifft2(kU0*Omega)), real(ifft2(kU1*Omega))
    ])
    u_mag = np.sqrt(u[0]**2 + u[1]**2)

    return u, u_mag

data = np.load("demo/data/" + savepath + "/v9 data s = 0.npz")
Omega = data['Omega']
P = data['P']
u, u_mag = calc_u(Omega)

fig, ax = plt.subplots(1, 1, constrained_layout = False)
u_mag_plot = ax.pcolormesh(x_grid, y_grid, u_mag, cmap = 'coolwarm')
u_vec_plot = ax.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo], u[0, ::eo, ::eo], u[1, ::eo, ::eo])
part_plot = plt.scatter(P[:, 0, 0], P[:, 0, 1], color = 'black')

ax.set_title(r"$\vec{u}(t$" + " = 0)")
ax.set_aspect('equal')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.xaxis.set_major_locator(MultipleLocator(base = np.pi))
ax.yaxis.set_major_locator(MultipleLocator(base = np.pi))

divider = make_axes_locatable(u_mag_plot.axes)
cax = divider.append_axes("right", size = "5%", pad = 0.2)
fig.colorbar(u_mag_plot, cax = cax, orientation = 'vertical')
cax.set_ylabel(r"$\mathrm{mag}\left[ \vec{u} \right]$")

data = np.load("demo/data/" + savepath + "/v9 data s = {0}.npz".format(6070))
t = data['t']
Omega = data['Omega']
P = data['P']
u, u_mag = calc_u(Omega)

u_mag_plot.set_array(u_mag)
u_vec_plot.set_UVC(u[0, ::eo, ::eo], u[1, ::eo, ::eo])
part_plot.set_offsets(np.stack((P[:, 0, 0], P[:, 0, 1]), axis = 1))
u_mag_plot.set_norm(Normalize(vmin = u_mag.min(), vmax = u_mag.max()))
ax.set_title(r"$\vec{u}(t = $" + "{0:.5f})".format(t))

plt.savefig("Vorticity-Steamfunction.png", dpi = 200)