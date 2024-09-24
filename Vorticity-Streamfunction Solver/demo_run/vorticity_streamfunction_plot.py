import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba
import matplotlib.pyplot as plt
import os

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

def save_particle_tracking(M, N, tau, s_max, s_int, savepath):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    eo = M // 32
    x_grid, y_grid = init_grid(M)[2:4]
    kU0, kU1 = init_wavearrays(M)[2:4]

    def calc_u(Omega):
        u = np.array([
            real(ifft2(kU0*Omega)), real(ifft2(kU1*Omega))
        ])
        u_mag = np.sqrt(u[0]**2 + u[1]**2)

        return u, u_mag
    
    data = np.load("data/" + savepath + "/v9 data s = 0.npz")
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

    def update_frame(s):
        data = np.load("data/" + savepath + "/v9 data s = {0}.npz".format(s))
        t = data['t']
        Omega = data['Omega']
        P = data['P']
        u, u_mag = calc_u(Omega)

        u_mag_plot.set_array(u_mag)
        u_vec_plot.set_UVC(u[0, ::eo, ::eo], u[1, ::eo, ::eo])
        part_plot.set_offsets(np.stack((P[:, 0, 0], P[:, 0, 1]), axis = 1))
        u_mag_plot.set_norm(Normalize(vmin = u_mag.min(), vmax = u_mag.max()))
        ax.set_title(r"$\vec{u}(t = $" + "{0:.5f})".format(t))

    ANI = FuncAnimation(fig, update_frame, init_func = lambda: None, frames = range(0, s_max, s_int)) # type: ignore
    ANI.save("anim/Inertial Particle Tracking Paths v9 {0} {1} {2}.mp4".format(M, N, tau), fps = 180, dpi = 200, writer = 'ffmpeg')

if __name__ == "__main__":
    # runtime parameters 
    M = 128 # square computational grid dimensions
    N = 50 # number of particles
    T = 150 # end time
    nu = 2e-3 # kinematic viscosity, see note above
    cfl_max = 0.7 # Courant–Friedrichs–Lewy Condition Number
    tau = 1 # inertial time
    beta = 1 # periods within the [0, 2π) domain
    gamma = 0.02 # perturbation magnitude
    forced = True
    rng_seed = 123

