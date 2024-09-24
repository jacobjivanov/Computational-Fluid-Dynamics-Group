import numpy as np
from numpy import real
from numpy.fft import ifft2, fft2
import matplotlib.pyplot as plt


M, N = 512, 512 # computational grid size
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]


def velocities(Omega):
    U, V = k_U * Omega, k_V * Omega
    u, v = real(ifft2(U)), real(ifft2(V))
    uv_mag = (u**2 + v**2) ** (1/2)

    return uv_mag, u, v

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))


    with np.errstate(divide = 'ignore', invalid = 'ignore'): # [p, q] = [0, 0] singularity
        k_U = +1j*k_q / (k_p**2 + k_q**2)
        k_V = -1j*k_p / (k_p**2 + k_q**2)
        k_Xi = k_p**2 + k_q**2
    k_U[0, 0], k_V[0, 0] = 0, 0

    return k_p, k_q, k_U, k_V

k_p, k_q, k_U, k_V = wavearrays(M, N)

current_file = np.load("Omega+Theta_p+t n = 28560.npz")
Omega = current_file['Omega']
omega = real(ifft2(Omega))
uv_mag, u, v = velocities(Omega)
U, V = fft2(u), fft2(v)
theta_p = real(ifft2(current_file['Theta_p']))
t = current_file['t']

diss = np.log10((real(ifft2(1j*k_p*U))**2 + real(ifft2(1j*k_p*V))**2 + real(ifft2(1j*k_q*U))**2 + real(ifft2(1j*k_q*V))**2)**2)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.pcolor(uv_mag/uv_mag.max())
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('uv_mag.png', dpi = 200)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.pcolor(diss/diss.max())
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('diss.png', dpi = 200)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.pcolor(theta_p/theta_p.max())
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('theta_p.png', dpi = 200)

fig, ax = plt.subplots(1, 1, figsize = (6, 6))
ax.pcolor(diss/diss.max() + 3*uv_mag/uv_mag.max())
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('comb.png', dpi = 200)