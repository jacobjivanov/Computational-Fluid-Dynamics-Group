import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba as nb

# simulation parameters
M, N = 1024, 1024 # computational grid size
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

n_p = 100
xp = np.random.rand(n_p) * 2 * np.pi
yp = np.random.rand(n_p) * 2 * np.pi

t, n = 0, 0
dt = 0.1
T = int(t_end / dt) + 1
psit0 = np.empty(shape = (T, n_p))
psit0[0] = psi_t0(xp, yp)

up = np.empty(n_p)
vp = np.empty(n_p)
up_star = np.empty(n_p)
vp_star = np.empty(n_p)
xp_star = np.empty(n_p)
yp_star = np.empty(n_p)

while t + dt < t_end:
    t += dt
    n += 1
    U *= np.exp(-nu * dt)
    V *= np.exp(-nu * dt)
    
    for p in range(0, n_p):
        # predictor step
        up[p] = real(inter_2D(U, xp[p], yp[p]))
        vp[p] = real(inter_2D(V, xp[p], yp[p]))

        xp_star[p] = xp[p] + dt * up[p] 
        yp_star[p] = yp[p] + dt * vp[p]

        # corrector step
        up_star[p] = real(inter_2D(np.exp(-nu * dt) * U, xp_star[p], yp_star[p]))
        vp_star[p] = real(inter_2D(np.exp(-nu * dt) * V, xp_star[p], yp_star[p]))

        xp[p] = (xp[p] + dt * (up[p] + up_star[p]) / 2) % (2 * np.pi)
        yp[p] = (yp[p] + dt * (vp[p] + vp_star[p]) / 2) % (2 * np.pi)

        psit0[n, p] = psi_t0(xp[p], yp[p])
    print(t)

import matplotlib.pyplot as plt
plt.plot(psit0 - psit0[0], color = 'blue')
plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.95))
plt.xlabel("timestep, $n$")
plt.ylabel(r"$\psi(t = 0, x_p(t), y_p(t)) - \psi_0$")
plt.title("Change in Streamfunction Value over Timestep")
plt.savefig("Non-Inertial Trajectory Tests/Predictor-Corrector TG {0} Trajectories.png".format(n_p), dpi = 200)