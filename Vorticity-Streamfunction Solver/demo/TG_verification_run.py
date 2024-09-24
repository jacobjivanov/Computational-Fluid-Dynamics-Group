# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant for Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. Contact me jacob.ivanov@uconn.edu for any questions/issues.
 
import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba
import matplotlib.pyplot as plt
import os

def init_grid(M):
    x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, M, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    dx, dy = x[1], y[1]

    return x, y, x_grid, y_grid, dx, dy

def init_wavearrays(M):
    kp = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1)).T

    kq = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1))
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        kU0 = +1j*kq / (kp**2 + kq**2)
        kU1 = -1j*kp / (kp**2 + kq**2)
        kXi = kp**2 + kq**2
    kU0[0, 0], kU1[0, 0] = 0, 0
    kU = np.array([kU0, kU1])
    kXi = nu * kXi

    return kp, kq, kU0, kU1, kU, kXi

def init_Omega():
    u_pert = 2*gamma * np.array([
        rng.random((M, M)) - 0.5,
        rng.random((M, M)) - 0.5
    ])

    u = np.array([
        +np.cos(x_grid) * np.sin(y_grid) + u_pert[0],
        -np.sin(x_grid) * np.cos(y_grid) + u_pert[1]
    ])

    U = np.array([fft2(u[0]), fft2(u[1])])
    Omega = 1j*(kp*U[1] - kq*U[0])
    dt = cfl_max / (np.amax(u[0])/dx + np.amax(u[1])/dy)

    return Omega, dt

@numba.njit()
def calc_tke(Omega):
    U = Omega * kU
    tke = np.sum(np.abs(U[0]) ** 2 + np.abs(U[1]) ** 2) # via Parseval's Theorem

    return tke

@numba.njit()
def calc_Xi(t):
    return np.exp(kXi * t)

@numba.njit()
def update_Omega(Omega0, dt):
    # RK3 Step Omega via XiOmega PDE explained here:
    # https://math.stackexchange.com/a/4835189/982890

    # Omega substep 1
    U0 = Omega0 * kU
    omega0 = ifft2(Omega0)
    u0 = np.zeros(shape = U0.shape)
    u0[0] = real(ifft2(U0[0]))
    u0[1] = real(ifft2(U0[1]))
    dXiOdt0 = -1j * (kp * fft2(u0[0] * omega0) + kq * fft2(u0[1] * omega0))
    
    # Omega substep 2
    Omega1 = calc_Xi(-8/15*dt) * (Omega0 + 8/15*dt*dXiOdt0)
    U1 = Omega1 * kU
    omega1 = ifft2(Omega1)
    u1 = np.zeros(shape = U1.shape)
    u1[0] = real(ifft2(U1[0]))
    u1[1] = real(ifft2(U1[1]))
    dXiOdt1 = -1j * calc_Xi(+8/15*dt) * (kp * fft2(u1[0] * omega1) + kq * fft2(u1[1] * omega1))

    # Omega substep 3
    Omega2 = calc_Xi(-2/3*dt) * (Omega0 + dt*(1/4*dXiOdt0 + 5/12*dXiOdt1))
    U2 = Omega2 * kU
    omega2 = ifft2(Omega2)
    u2 = np.zeros(shape = U2.shape)
    u2[0] = real(ifft2(U2[0]))
    u2[1] = real(ifft2(U2[1]))
    dXiOdt2 = -1j * calc_Xi(+2/3*dt) * (kp * fft2(u2[0] * omega2) + kq * fft2(u2[1] * omega2))

    # Omega combinatory substep
    Omega3 = calc_Xi(-dt) * (Omega0 + dt*(1/4*dXiOdt0 + 3/4*dXiOdt2))
    Omega3[M//3:2*M//3+1, M//3:2*M//3+1] = 0 # removal of high-frequency vorticity
    U3 = Omega3 * kU

    u3 = np.zeros(shape = U3.shape)
    u3[0] = real(ifft2(U3[0]))
    u3[1] = real(ifft2(U3[1]))
    dt_next = cfl_max / (np.amax(u3[0])/dx + np.amax(u3[1])/dy)

    return Omega3, U3, dt_next

# NOTE: Recommended Viscosities for Well-Resolved Flow
"""
M   |   nu
64  |   5e-3
128 |   2e-3
256 |   9e-4
512 |   2e-4
1024|   8e-5
"""

if __name__ == "__main__":
    # runtime parameters 
    M = 256 # square computational grid dimensions
    N = 50 # number of particles
    T = 150 # end time
    nu = 9e-4 # kinematic viscosity, see note above
    cfl_max = 0.7 # Courant–Friedrichs–Lewy Condition Number
    tau = 1 # inertial time
    gamma = 0 # perturbation magnitude
    rng_seed = 123

    # initialization
    rng = np.random.default_rng(seed = rng_seed)
    x, y, x_grid, y_grid, dx, dy = init_grid(M)
    kp, kq, kU0, kU1, kU, kXi = init_wavearrays(M)
    Omega, dt = init_Omega()
    savepath = "{0} {1}".format(M, gamma)
    if not os.path.exists("TG_verification/" + savepath): os.mkdir("TG_verification/" + savepath)

    t = 0 # time
    s = 0 # step
    s_int = 5 # step/save interval
    while t < T:
        if s % s_int == 0:
            tke = calc_tke(Omega)
            print("n = {0:6}, t = {1:.5f}, tke = {2:.5e}".format(s, t, tke))
            np.savez("TG_verification/" + savepath + "/s = {0}.npz".format(s), t = t, Omega = Omega)  # type: ignore

        Omega, U, dt_next = update_Omega(Omega, dt)
        t += dt
        s += 1

        dt = dt_next

    s_max = s
