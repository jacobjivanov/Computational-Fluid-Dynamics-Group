# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# The purpose of this program is to implement Stabilized Turbulent Kinetic Energy (STKE) Spectral Vorticity-Streamfunction Evolution

import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2
from numpy.random import rand

import numba as nb
import matplotlib.pyplot as plt

from STKE_helpers import wavearrays

# runtime dependent parameters
M, N = 64, 64 # computational grid dimensions
nu = 5e-3 # kinematic viscosity
beta = 1 # periods within the [0, 2π) domain
gamma = 0.02 # perturbation magnitude
t_end = 200
cfl_max = 0.8

# boilerplate initialization
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
dx, dy = x[1], y[1]
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
k_p, k_q, k_U, k_V = wavearrays(M, N)
k_Xi = k_p**2 + k_q**2
k_Xi_f = k_p**2 + k_q**2

# forced turbulence viscosity modification
for p in range(0, M):
    for q in range(0, N):
        if 5**2 <= k_Xi_f[p, q] and k_Xi_f[p, q] <= 6**2:
            k_Xi_f[p, q] *= -1
        if k_Xi_f[p, q] <= 2**2:
            k_Xi_f[p, q] *= 8

x_pert = 2 * gamma * (rand(M, N) - 0.5) # initial x-velocity perturbation
y_pert = 2 * gamma * (rand(M, N) - 0.5) # initial x-velocity perturbation
u = +np.cos(beta*x_grid) * np.sin(beta*y_grid) + x_pert # physcial velocity x-component
v = -np.sin(beta*x_grid) * np.cos(beta*y_grid) + y_pert # physical velocity y-component

U, V = fft2(u), fft2(v)
Omega = 1j*(k_p*V - k_q*U)
omega = real(ifft2(Omega))

# main loop functions
@nb.njit()
def Xi(t, TKE):
    if TKE < TKE0:
        return np.exp(nu * k_Xi_f * t)
    else:
        return np.exp(nu * k_Xi * t)

@nb.njit()
def pXiOmegapt(t, XiOmega, TKE):
    Omega = Xi(-t, TKE) * XiOmega
    omega = ifft2(Omega)

    U, V = k_U * Omega, k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))

    pXiOmegapt = -1j * Xi(t, TKE) * (k_p * fft2(u * omega) + k_q * fft2(v * omega))
    dt = cfl_max / (np.amax(u)/dx + np.amax(v)/dy)

    return pXiOmegapt, dt

@nb.njit()
def Omega_RK3step(dt, Omega, TKE):
    k1, dt_next = pXiOmegapt(
        0, 
        Omega, TKE)
    k2, dt_next = pXiOmegapt(
        dt * 8/15,
        Omega + dt * 8/15 * k1, TKE)
    k3, dt_next = pXiOmegapt(
        dt * 2/3,
        Omega + dt * (1/4 * k1 + 5/12 * k2), TKE)

    Omega += dt * (1/4 * k1 + 3/4 * k3)
    Omega *= Xi(-dt, TKE)
    Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0

    U, V = k_U * Omega, k_V * Omega
    TKE = np.sum(np.abs(U) ** 2 + np.abs(V) ** 2)
    return Omega, dt_next, TKE

t, n = 0, 0
TKE0 = np.sum(np.abs(U) ** 2 + np.abs(V) ** 2)
TKE = TKE0
dt_next = pXiOmegapt(0, Omega, TKE)[1]

while t < t_end:
    dt = dt_next
    
    if n % 10 == 0:
        print("{0}, {1:.5f}, {2:.5f}, {3:.5e}".format(n, dt, t, TKE))
        np.savez("STKE Vorticity Evolution/temp/STKE v0 64/Omega+t n = {0}.npz".format(n), Omega = Omega, t = t)

    Omega, dt_next, TKE = Omega_RK3step(dt, Omega, TKE)

    t += dt
    n += 1