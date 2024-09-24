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
gamma = 0 #0.02 # perturbation magnitude
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
cfl_max = 0.8
n = 0

def f(Omega):
    U = k_U * Omega
    V = k_V * Omega

    Omega_x = 1j * k_p * Omega
    Omega_y = 1j * k_q * Omega

    u = real(ifft2(U))
    v = real(ifft2(V))

    omega_x = real(ifft2(Omega_x))
    omega_y = real(ifft2(Omega_y))

    dt = cfl_max / (np.amax(u)/dx + np.amax(v)/dy)

    print(np.amax(u), np.amax(v))

    return -fft2(u * omega_x + v * omega_y), dt

def Xi(t):
    return nu * k_Xi * t

Omega_3 = Omega
dt = f(Omega_3)[1]

while t < t_end:
    if n % 1 == 0:
        # print(t, dt)
        a = 1

    Omega_0 = Omega_3
    f0 = f(Omega_0)[0]

    Omega_1 = Xi(- 8/15 * dt) * (Omega_0 + dt * 8/15 * f0)
    f1 = f(Omega_1)[0]

    plt.pcolormesh(real(ifft2(Omega_1)))
    plt.colorbar()
    plt.show()

    Omega_2 = Xi(- 2/15 * dt) * (Omega_1 + dt * (-17/60 * f0 * Xi(- 8/15 * dt) + 5/12 * f1))
    f2, dt = f(Omega_2)
    # print(dt)
    Omega_3 = Xi(- 1/3 * dt) * (Omega_2 + dt * (-5/12 * f1 * Xi(- 2/15 * dt) + 3/4 * f2))
    
    Omega_3[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0

    # print("a")
    n += 1
    t += dt