import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numba as nb

# computational grid
M, N = 128, 128
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

# fluid parameters
nu = 1e-3

n_part = 5 # number of tracked particles
x_part = np.random.rand(n_part) * 2 * np.pi
y_part = np.random.rand(n_part) * 2 * np.pi

# defining various wavenumber constants
def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

k_p, k_q = wavearrays(M, N)
with np.errstate(divide = 'ignore', invalid = 'ignore'): # [p, q] = [0, 0] singularity
    k_Psi = 1 / (k_p**2 + k_q**2)
    k_U = +1j*k_q / (k_p**2 + k_q**2)
    k_V = -1j*k_p / (k_p**2 + k_q**2)
    k_Xi = k_p**2 + k_q**2
k_Psi[0, 0], k_U[0, 0], k_V[0, 0] = 0, 0, 0

for i in range(0, M):
    for j in range(0, N):
        if 6**2 <= k_Xi[i, j] and k_Xi[i, j] <= 7**2:
            k_Xi[i, j] *= -1
        if k_Xi[i, j] <= 2**2:
            k_Xi[i, j] *= 8

# initial conditions
beta = 1 # periods within the [0, 2π) domain
gamma = 0.02 # perturbation magnitude
x_pert = 2*gamma * (np.random.rand(M, N) - 0.5) # initial x-velocity perturbation
y_pert = 2*gamma * (np.random.rand(M, N) - 0.5) # initial x-velocity perturbation
u = +np.cos(beta*x_grid) * np.sin(beta*y_grid) + x_pert
 # physcial velocity x-component
v = -np.sin(beta*x_grid) * np.cos(beta*y_grid) + y_pert# physical velocity y-component

U, V = fft2(u), fft2(v)
Omega = 1j*(k_p*V - k_q*U)
omega = real(ifft2(Omega))
Psi = k_Psi * Omega
psi = real(ifft2(Psi))

"""
def TG_flow(t):
    u = np.exp(-2*nu*t) * np.cos(beta*x_grid) * np.sin(beta*y_grid)
    v = - np.exp(-2*nu*t) * np.sin(beta*x_grid) * np.cos(beta*y_grid)
    omega = -2*beta * np.exp(-2*nu*t) * np.cos(beta*x_grid) * np.cos(beta*y_grid)
    psi = - np.exp(-2*nu*t) * np.cos(beta * x_grid) * np.cos(beta * y_grid) / beta
    
    return u, v, psi, omega
"""

def omega_TG(t): # analytical Taylor-Greene Vorticity
    omega_TG = -2*beta * np.exp(-2*nu*t) * np.cos(beta*x_grid) * np.cos(beta*y_grid)
    return omega_TG

@nb.njit()
def inter_2D(kp, kq, U, pos):
    M, N = U.shape

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, N):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * pos[0])

    u_inter /= M
    return u_inter

@nb.njit()
def pXiOmegapt(t, XiOmega):
    Xi = np.exp(nu * k_Xi * t)
    Omega = XiOmega / Xi
    
    U, V = k_U * Omega, k_V * Omega
    omega, u, v = real(ifft2(Omega)), real(ifft2(U)), real(ifft2(V))
    UOmega, VOmega = fft2(u * omega), fft2(v * omega)

    pXiOmegapt = -1j * Xi * (k_p * UOmega + k_q * VOmega)
    pXiOmegapt[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

    return pXiOmegapt, np.amax(u), np.amax(v), U, V

XiOmega = Omega
t, t_end = 0, 200
cfl_max = 0.8
n = 0

u_max, v_max = np.amax(u), np.amax(v)

while t < t_end:
    print("{0}, {1:.3f}".format(n, t))
    if n % 5 == 0:
        Xi = np.exp(nu * k_Xi * t)
        Omega = XiOmega / Xi
        U, V = k_U * Omega, k_V * Omega
        omega, u, v = real(ifft2(Omega)), real(ifft2(U)), real(ifft2(V))
        uv = (u ** 2 + v ** 2) ** 0.5
        
        eo = M // 32
        fig, ax = plt.subplots(1, 1, constrained_layout = False)
        uv_plot = ax.pcolormesh(x_grid, y_grid, uv, cmap = 'coolwarm')
        
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

        ax.quiver(x[::eo], y[::eo], u[::eo, ::eo], v[::eo, ::eo], pivot = 'mid', color = 'grey', alpha = 0.5)
        ax.scatter(x_part, y_part, marker = 'x', color = 'black')
        ax.set_title(r"$\vec{u}(t = $" + "{0:5f})".format(t))
        
        # plt.savefig("v4_plots/velocity field t = {:.5f}.png".format(t), dpi = 300, bbox_inches = 'tight')
        # plt.clf()
        # plt.close()
        plt.show()
    
    dt = cfl_max / (u_max/dx + v_max/dy)

    k_1, u_max, v_max, U, V = pXiOmegapt(t, XiOmega)
    k_2, u_max, v_max, U, V = pXiOmegapt(t + 8/15 * dt, XiOmega + dt * 8/15 * k_1)
    k_3, u_max, v_max, U, V = pXiOmegapt(t + 2/3 * dt, XiOmega + dt * (1/4 * k_1 + 5/12 * k_2))

    XiOmega += dt * (1/4 * k_1 + 3/4 * k_3)
    
    for p in range(0, n_part):
        x_part[p] = (x_part[p] - dt * real(inter_2D(k_p, k_q, V, [x_part[p], y_part[p]]))) % (2 * np.pi)
        y_part[p] = (y_part[p] - dt * real(inter_2D(k_p, k_q, U, [x_part[p], y_part[p]]))) % (2 * np.pi)

    t += dt
    n += 1