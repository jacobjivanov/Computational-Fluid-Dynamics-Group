import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

import numba

@numba.njit()
def psi_t0(x, y):
    return -1/beta * np.cos(beta * x) * np.cos(beta * y)

def wavearrays():
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1)).T

    k_q = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

@numba.njit()
def calc_inter2D(U, x, y):
    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, M):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y)

        u_yinter /= M
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x)

    u_inter /= M
    return u_inter

def init_grid(M):
    x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, M, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    dx, dy = x[1], y[1]

    return x, y, x_grid, y_grid, dx, dy

def init_P():
    P = np.zeros(shape = (N, 4, 2))
    P[:, 0, :] = rng.random((N, 2)) * 2 * np.pi
    
    # assuming particles are just dropped into flow, with no initial velocity, but will have fluid velocity  equal to that of the surrounding flow

    for n in range(0, N):
        P[n, 2, :] = np.array([
            + np.cos(beta*P[n, 0, 0]) * np.sin(beta*P[n, 0, 1]),
            - np.sin(beta*P[n, 0, 0]) * np.cos(beta*P[n, 0, 1])
        ])
        
        P[n, 3, :] = 0
    
    return P

def init_U():
    u = np.array([
        + np.cos(x_grid) * np.sin(y_grid), 
        - np.sin(x_grid) * np.cos(y_grid)
    ])

    U = np.array([
        fft2(u[0]),
        fft2(u[1]),
    ])

    return U

@numba.njit(parallel = True)
def update_P(P0, U3, dt):
    # Direct Heun's Method, updating particle position via surrounding fluid velocity, which the above method degenerates to for non-inertial particles, ie where `tau` is zero
    Pp = np.empty(shape = P0.shape, dtype = np.float64)
    Pc = np.empty(shape = P0.shape, dtype = np.float64)

    for n in numba.prange(0, N): # will auto-parallelize each particle
        # inertial particles predictor substep
        Pp[n, 0] = P0[n, 0] + dt*P0[n, 2]

        Pp[n, 2, 0] = real(calc_inter2D(U3[0], Pp[n, 0, 0], Pp[n, 0, 1]))
        Pp[n, 2, 1] = real(calc_inter2D(U3[1], Pp[n, 0, 0], Pp[n, 0, 1]))

        Pc[n, 2] = Pp[n, 2]
        Pc[n, 0] = P0[n, 0] + dt*(P0[n, 2] + Pp[n, 2])/2
    
    return Pc

rng = np.random.default_rng(seed = 123)

if __name__ == "__main__":
    # simulation parameters
    M = 256 # computational grid size
    nu = 5e-3 # kinematic viscosity
    beta = 1 # periods within the [0, 2π) domain
    t_end = 5

    x, y, x_grid, y_grid, dx, dy = init_grid(M)
    kp, kq = wavearrays()
    fig, ax = plt.subplots(figsize = (7.5, 3))
    ax.set_xscale("log")
    ax.set_yscale("log")

    DT = np.array([0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1])
    for dt in DT:
        N = 10
        P = init_P()
        U = init_U()

        t, n = 0, 0
        T = int(t_end / dt) + 1
        psit0 = np.empty(shape = (T, N))
        psit0[0] = psi_t0(P[:, 0, 0], P[:, 0, 1])

        while t + dt < t_end:
            t += dt
            n += 1
            U *= np.exp(-nu * dt)
            P = update_P(P, U, dt)
            
            psit0[n] = psi_t0(P[:, 0, 0], P[:, 0, 1])
            print("{0:.5f}".format(t), np.max(np.abs(psit0[n] - psit0[0])))

        ax.scatter(np.zeros(N) + dt, np.abs(psit0[T-2] - psit0[0]), color = 'blue', alpha = 0.1)
    
    ax.plot(DT, 1e-3 * DT**2, label = '2nd Order Convergence', color = 'red', linestyle = 'dashed')

    fig.tight_layout(rect = (0.05, 0.05, 0.95, 0.95))
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|\psi(t = 0, \vec{x}(t)) - \psi(t = 0, \vec{x}(0))|$")
    ax.set_title("Streamline Deviation")
    plt.legend()
    plt.savefig("predictor-corrector_verification.png".format(N), dpi = 200)
    plt.show()