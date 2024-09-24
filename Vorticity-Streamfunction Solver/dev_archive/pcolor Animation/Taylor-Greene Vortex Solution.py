import numpy as np
from numpy.fft import fft2
import numba as nb

# simulation parameters
M, N = 128, 128 # computational grid size
nu = 5e-3 # kinematic viscosity
beta = 1 # periods within the [0, 2π) domain
t_end = 200
dt = 0.01

@nb.njit()
def Omega_TG(t): # analytical Taylor-Greene Vorticity
    omega_TG = -2*beta * np.exp(-2*nu*t) * np.cos(beta*x_grid) * np.cos(beta*y_grid)
    return fft2(omega_TG)

# computational grid
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]

t, n = 0, 0
while t < t_end:
    print("{0}, {1:.5f}".format(n, t))
    Omega_ana = Omega_TG(t)
    if n % 10 == 0:
        np.save("pcolor Animation/temp/Omega_ana, n = {}".format(n), Omega_ana)

    n += 1
    t += dt