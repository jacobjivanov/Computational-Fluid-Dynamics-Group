import numpy as np
import numba as nb
import pandas as pd

import matplotlib.pyplot as plt

# simulation parameters
M = 128 # computational grid size
nu = 5e-3 # kinematic viscosity
beta = 1 # periods within the [0, 2π) domain
T = 100 # end time
dt = 0.1 # time step
N = 100 # particle count
tau = 0.1 # particle inertial time

x = np.linspace(0, 2*np.pi, M, endpoint = False)
y = np.linspace(0, 2*np.pi, M, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]
eo = M // 32

f = open('inertial_particles/particle_positions.csv', 'w')
f.write("t,")
for n in range(0, N):
    f.write("x{0},y{0},".format(n))
f.write("\n")

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

kp, kq = wavearrays(M, N)

# @nb.njit()
def u_grid(t):
    return np.exp(-nu * t) * np.array([np.cos(beta*x_grid) * np.sin(beta*y_grid), -np.sin(beta*x_grid) * np.cos(beta*y_grid)])

# @nb.njit()
def a_grid(t):
    return -nu * np.exp(-nu * t) * np.array([np.cos(beta*x_grid) * np.sin(beta*y_grid), -np.sin(beta*x_grid) * np.cos(beta*y_grid)])

@nb.njit()
def psi_t0(x, y):
    return -1/beta * np.cos(beta * x) * np.cos(beta * y)

@nb.njit()
def inter_2D(U, x, y):
    M = U.shape[0]

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, M):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y)

        u_yinter /= M
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x)

    u_inter /= M
    return u_inter

u = u_grid(t = 0)

U = np.empty(shape = (2, M, M), dtype = 'complex')
U[0] = np.fft.fft2(u[0])
U[1] = np.fft.fft2(u[1])

xp = np.random.rand(N, 2) * 2 * np.pi
xp_star = np.empty(shape = (N, 2))
up = np.empty(shape = (N, 2))
up_star = np.empty(shape = (N, 2))
vp = np.empty(shape = (N, 2))
vp_star = np.empty(shape = (N, 2))
ap = np.empty(shape = (N, 2))
ap_next = np.empty(shape = (N, 2))

t = 0
while t < T:
    U *= np.exp(-nu * dt)
    psit0 = psi_t0(xp[0, 0], xp[0, 1])

    f.write("{0},".format(t))
    for n in range(0, N):
        if tau != 0:
            # predictor substep
            xp_star[n] = xp[n] + dt*vp[n]
            up_star[n, 0] = np.real(inter_2D(np.exp(-nu * dt) * U[0], *xp[n]))
            up_star[n, 1] = np.real(inter_2D(np.exp(-nu * dt) * U[1], *xp[n]))
            ap[n] = -nu * np.exp(nu * dt) * up_star[n]

            vp_star[n, 0] = vp[n, 0] + dt*(up_star[n, 0]/tau + 3*ap[n, 0])
            vp_star[n, 1] = vp[n, 1] + dt*(up_star[n, 1]/tau + 3*ap[n, 1])

            # corrector substep
            xp[n] = xp[n] + dt/2 * (vp[n] + vp_star[n])
            up_star[n, 0] = np.real(inter_2D(np.exp(-nu * dt) * U[0], *xp[n]))
            up_star[n, 1] = np.real(inter_2D(np.exp(-nu * dt) * U[1], *xp[n]))
            ap_next[n] = -nu * up[n]
            
            vp[n] = (2*tau)/(2*tau + dt) * ((1 - dt/(2*tau))*vp[n] + dt/2 * (3*(ap[n] + ap_next[n]) + (up[n] + up_star[n])/tau))

            up[n] = up_star[n]

        if tau == 0:
            # predictor substep
            vp[n, 0] = np.real(inter_2D(U[0], *xp[n]))
            vp[n, 1] = np.real(inter_2D(U[1], *xp[n]))
            xp_star[n] = xp[n] + dt*vp[n]

            # corrector substep
            vp_star[n, 0] = np.real(inter_2D(np.exp(-nu * dt) * U[0], *xp_star[n]))
            vp_star[n, 1] = np.real(inter_2D(np.exp(-nu * dt) * U[1], *xp_star[n]))
            
            xp[n] = xp[n] + dt/2 * (vp[n] + vp_star[n])
        
        xp[n] = xp[n] % (2 * np.pi)

        # print("psi_t0: {0:.5e}".format(psi_t0(xp[n, 0], xp[n, 1]) - psit0))

        f.write("{0},{1},".format(xp[n, 0], xp[n, 1]))
    f.write("\n")

    t += dt
    print("{0:.5f}".format(t), end = '\r')

plt.show()