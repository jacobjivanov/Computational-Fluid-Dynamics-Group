# vorticity_evolution_v3.py

import numpy as np
from numpy import real
from numpy.fft import fft2, ifft2
from numpy.random import rand
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import rocket_fft
rocket_fft.numpy_like()

M, N = 64, 64
x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
dx, dy = x[1] - x[0], y[1] - y[0]

# defining various wavenumber constants
def wavearrays(M, N, x_dom, y_dom):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))) * 2 * np.pi / (x_dom[1] - x_dom[0]), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (y_dom[1] - y_dom[0]), reps = (M, 1))

    return k_p, k_q

k_p, k_q = wavearrays(M, N, x_dom, y_dom)
with np.errstate(divide = 'ignore', invalid = 'ignore'): # singularity where p = q = 0
    k_U = + 1j * k_q / (k_p ** 2 + k_q ** 2)
    k_V = - 1j * k_p / (k_p ** 2 + k_q ** 2)
    k_Xi = k_p ** 2 + k_q ** 2
k_U[0, 0], k_V[0, 0] = 0, 0

# fluid parameters
nu = 3e-4 # kinematic viscosity
ar = 0.0 # magnitude of initial perturbations
sc = 0.7 # schidt number
b = 0.2 # scalar gradient
per = 1 # periods within domain

# initial conditions
u = np.empty(shape = (M, N)) # physcial velocity x-component
v = np.empty(shape = (M, N)) # physical velocity y-component
for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(per * x[i]) * np.sin(per * y[j]) + 2 * ar * (rand() - 0.5)
        v[i, j] = - np.sin(per * x[i]) * np.cos(per * y[j]) + 2 * ar * (rand() - 0.5)

U, V = np.fft.fft2(u), np.fft.fft2(v)

Omega = np.empty(shape = (M, N), dtype = 'complex')
Omega = 1j * (k_p * V - k_q * U)

Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

@nb.njit()
def pXiOmegapt(t, XiOmega):
    Xi = np.exp(nu * k_Xi * t)
    
    Omega = XiOmega / Xi
    U, V = k_U * Omega, k_V * Omega
    omega, u, v = ifft2(Omega), ifft2(U), ifft2(V)

    UOmega, VOmega = fft2(u * omega), fft2(v * omega)
    
    pXiOmegapt = -1j * Xi * (k_p * UOmega + k_q * VOmega)
    pXiOmegapt[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency voriticity

    return pXiOmegapt, omega, real(u), real(v)


# tested various Butcher Tables, but Vorticity Evolution goes unstable t ~ 90 regardless
a_RK3 = np.array([[8/15, 0], [1/4, 5/12]])
b_RK3 = np.array([1/4, 0, 3/4])
c_RK3 = np.array([0, 8/15, 2/3])

a_RK4 = np.array([[1/2, 0, 0], 
              [0, 1/2, 0], 
              [0, 0, 1]])
b_RK4 = np.array([1/6, 1/3, 1/3, 1/6])
c_RK4 = np.array([0, 1/2, 1/2, 1])

a_ralston = np.array([
    [0.4, 0, 0],
    [0.29697761, 0.15875964, 0],
    [0.21810040, -3.05096516, 3.83286476]
])
b_ralston = np.array([0.17476028, -0.55148066, 1.20553560, 0.17118478])
c_ralston = np.array([0, 0.4, 0.45573725, 1])

@nb.njit()
def rk_step(f, t0, y0, dt, a = a_RK3, b = b_RK3, c = c_RK3):
    s = c.size
    assert s > 0

    k = np.zeros(shape = (s, *y0.shape), dtype = 'complex')
    for rk_i in range(0, s):
        y_substep = y0
        for rk_j in range(0, rk_i):
            y_substep += a[rk_i - 1, rk_j] * k[rk_j] * dt
        t_substep = t0 + c[rk_i] * dt
        k[rk_i::], omega, u, v = f(t_substep, y_substep)
    
    y_step = y0
    for rk_j in range(0, s):
        y_step += b[rk_j] * k[rk_j] * dt

    return y_step, omega, u, v # type: ignore[unbound]

# computational time-step parameters
cfl_max = 0.5
t, t_end = 0, 200
n = 0

XiOmega = Omega
while t < t_end:
    if n % 10 == 0:
        uv = (u ** 2 + v ** 2) ** 0.5
        
        fig, ax = plt.subplots(1, 3, constrained_layout = True, figsize = (8, 5))
        fig.suptitle("\n\nVorticity-Streamfunction Evolution\n{0}% Perturbed Taylor-Greene Vortex Flow\nt = {1:.5f}, n = {2}".format(ar * 100, t, n))

        eo = M // 16
        ax[0].set_title(r"$\vec{u}$")
        vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
        ax[0].quiver(x[::eo], y[::eo], u[::eo, ::eo].T, v[::eo, ::eo].T, pivot = 'mid')
        ax[0].set_aspect('equal')
        
        dil = np.zeros(shape = (M, N))

        ax[1].set_title(r"$\nabla \cdot \vec{u}$, NOT ACTIVE")
        ax[1].set_aspect('equal')
        dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

        phi = np.zeros(shape = (M, N))

        ax[2].set_title(r"$\phi$, NOT ACTIVE")
        ax[2].set_aspect('equal')
        phi_plot = ax[2].pcolormesh(x, y, np.transpose(phi), cmap = 'coolwarm')

        for plt_i in [0, 1, 2]:
            ax[plt_i].set_xlim(0, 2 * np.pi)
            ax[plt_i].set_ylim(0, 2 * np.pi)

            ax[plt_i].set_xlabel(r"$x$")
            ax[plt_i].set_ylabel(r"$y$")

            ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
            ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = np.pi))

        for plot in [vel_plot, dil_plot, phi_plot]:
            divider = make_axes_locatable(plot.axes)
            cax = divider.append_axes("bottom", size = "5%", pad = 0.5)
            fig.colorbar(plot, cax = cax, orientation = 'horizontal')

        plt.show()

    dt = cfl_max / (np.amax(u) / dx + np.amax(v) / dy)
    XiOmega, omega, u, v = rk_step(pXiOmegapt, t, XiOmega, dt)
    
    n += 1
    if n % 1 == 0:
        print("t = {0:.5f}, dt = {1:.5f} n = {2}".format(t, dt, n))

    t += dt