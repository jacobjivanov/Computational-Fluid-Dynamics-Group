import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba as nb
import matplotlib.pyplot as plt

M, N = 64, 64
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

# fluid parameters
nu = 5e-3

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
        if 5**2 <= k_Xi[i, j] and k_Xi[i, j] <= 6**2:
            k_Xi[i, j] *= -1
        if k_Xi[i, j] <= 2**2:
            k_Xi[i, j] *= 8


@nb.njit()
def Xi(t):
    return np.exp(nu * k_Xi * t)

@nb.njit()
def pXiOmegapt(t, XiOmega):
    Omega = Xi(-t) * XiOmega
    omega = ifft2(Omega)

    U, V = k_U * Omega, k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))
    
    pXiOmegapt = -1j * Xi(t) * (k_p * fft2(u * omega) + k_q * fft2(v * omega))
    dt = cfl_max / (np.amax(u)/dx + np.amax(v)/dy)
    return pXiOmegapt, dt

@nb.njit()
def omega_TG(t): # analytical Taylor-Greene Vorticity
    omega_TG = -2*beta * np.exp(-2*nu*t) * np.cos(beta*x_grid) * np.cos(beta*y_grid)
    return omega_TG

# initial conditions
beta = 1 # periods within the [0, 2π) domain
gamma = 0 # 0.02 # perturbation magnitude
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

@nb.njit()
def Omega_3step(dt, Omega):
    k1, dt_next = pXiOmegapt(
        0, 
        Omega)
    k2, dt_next = pXiOmegapt(
        dt * 8/15,
        Omega + dt * 8/15 * k1)
    k3, dt_next = pXiOmegapt(
        dt * 2/3,
        Omega + dt * (1/4 * k1 + 5/12 * k2))

    Omega += dt * (1/4 * k1 + 3/4 * k3)
    Omega *= Xi(-dt)
    Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0

    return Omega, dt_next

t, t_end = 0, 200
cfl_max = 0.5
n = 0

dt_next = pXiOmegapt(0, Omega)[1]

while t < t_end:
    dt = dt_next
    
    if n % 10 == 0:
        print("{0}, {1:.5f}, {2:.5f}".format(n, dt, t))
        
        U, V = k_U * Omega, k_V * Omega
        u, v = real(ifft2(U)), real(ifft2(V))
        uv_mag = np.sqrt(u**2 + v**2)
        
        plt.pcolormesh(x_grid, y_grid, uv_mag, cmap = 'coolwarm')
        plt.colorbar()
        eo = M // 16
        plt.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo], u[::eo, ::eo], v[::eo, ::eo], pivot = 'mid')
        plt.show()
    
    Omega, dt_next = Omega_3step(dt, Omega)

    t += dt
    n += 1