# the following pretty much copies the MATLAB section while t < t_end loop

import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import matplotlib.pyplot as plt

M, N = 256, 256
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

# fluid parameters
nu = 9e-4

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

t, t_end = 0, 200
cfl_max = 0.7
n = 0

while t < t_end:
    dt = cfl_max / (np.amax(u)/dx + np.amax(v)/dy)
    
    if n % 200 == 0:
        print(n, dt, t)
        plt.pcolormesh(real(ifft2(Omega)))
        plt.colorbar()
        plt.show()

    # substep 1
    U = k_U * Omega
    V = k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))

    Omega_x = 1j * k_p * Omega
    Omega_y = 1j * k_q * Omega

    omega_x = real(ifft2(Omega_x))
    omega_y = real(ifft2(Omega_y))

    f0 = - fft2(u * omega_x + v * omega_y)

    Xi = np.exp(-nu * k_Xi * 8/15 * dt)
    Omega = Xi * (Omega + dt * 8/15 * f0)

    # substep 2
    U = k_U * Omega
    V = k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))

    Omega_x = 1j * k_p * Omega
    Omega_y = 1j * k_q * Omega

    omega_x = real(ifft2(Omega_x))
    omega_y = real(ifft2(Omega_y))

    f1 = - fft2(u * omega_x + v * omega_y)
    Omega = Omega + dt * (-17/60 * Xi * f0 + 5/12 * f1)
    Xi = np.exp(- nu * k_Xi * 2/15 * dt)
    Omega = Xi * Omega

    # substep 3
    U = k_U * Omega
    V = k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))

    Omega_x = 1j * k_p * Omega
    Omega_y = 1j * k_q * Omega

    omega_x = real(ifft2(Omega_x))
    omega_y = real(ifft2(Omega_y))

    f2 = - fft2(u * omega_x + v * omega_y)
    Omega = Omega + dt * (-5/12 * Xi * f1 + 3/4 * f2)
    Xi = np.exp(- nu * k_Xi * 1/3 * dt)
    Omega = Xi * Omega

    t += dt
    n += 1