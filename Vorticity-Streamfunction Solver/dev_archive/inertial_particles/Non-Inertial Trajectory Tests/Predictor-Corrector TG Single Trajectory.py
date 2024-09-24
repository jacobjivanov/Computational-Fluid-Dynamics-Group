import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba as nb

# simulation parameters
M, N = 128, 128 # computational grid size
nu = 5e-3 # kinematic viscosity
beta = 1 # periods within the [0, 2π) domain
t_end = 200

x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]
eo = M // 32

u = + np.cos(beta * x_grid) * np.sin(beta * y_grid)
v = - np.sin(beta * x_grid) * np.cos(beta * y_grid)
U, V = fft2(u), fft2(v)

@nb.njit()
def psi_t0(x, y):
    return -1/beta * np.cos(beta * x) * np.cos(beta * y)

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

kp, kq = wavearrays(M, N)

@nb.njit()
def inter_2D(U, x, y):
    M, N = U.shape

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, N):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y)

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x)

    u_inter /= M
    return u_inter

xp = np.random.rand() * 2 * np.pi
yp = np.random.rand() * 2 * np.pi

t, n = 0, 0
dt = 0.1
T = int(t_end / dt) + 1
psit0 = np.empty(shape = (T))
psit0[0] = psi_t0(xp, yp)

while t + dt < t_end:
    t += dt
    n += 1
    U *= np.exp(-nu * dt)
    V *= np.exp(-nu * dt)
    
    # predictor step
    up = real(inter_2D(U, xp, yp))
    vp = real(inter_2D(V, xp, yp))

    xp_star = xp + dt * up 
    yp_star = yp + dt * vp

    # corrector step
    up_star = real(inter_2D(np.exp(-nu * dt) * U, xp_star, yp_star))
    vp_star = real(inter_2D(np.exp(-nu * dt) * V, xp_star, yp_star))

    xp = (xp + dt * (up + up_star) / 2) % (2 * np.pi)
    yp = (yp + dt * (vp + vp_star) / 2) % (2 * np.pi)

    psit0[n] = psi_t0(xp, yp)
    print(t)

import matplotlib.pyplot as plt
plt.plot(psit0 - psit0[0], color = 'blue')
plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.95))
plt.xlabel("timestep, $n$")
plt.ylabel(r"$\psi(t = 0, x_p(t), y_p(t)) - \psi_0$")
plt.title("Change in Streamfunction Value over Timestep")
plt.show()