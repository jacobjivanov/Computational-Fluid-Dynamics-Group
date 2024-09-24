# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant for Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. Contact me jacob.ivanov@uconn.edu for any questions/issues.

# The goal of this program is to evolve the vorticity transport equation initialized as perturbed Taylor-Green Vortex Flow. Additionally, variably inertial particles will be trasported within the flow.

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

    # forced turbulence viscosity modification
    if forced == True:
        for p in range(0, M):
            for q in range(0, M):
                if 5**2 <= kXi[p, q] and kXi[p, q] <= 6**2:
                    kXi[p, q] *= -1
                if kXi[p, q] <= 2**2:
                    kXi[p, q] *= 8
    
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
        +np.cos(beta*x_grid) * np.sin(beta*y_grid) + u_pert[0],
        -np.sin(beta*x_grid) * np.cos(beta*y_grid) + u_pert[1]
    ])

    U = np.array([fft2(u[0]), fft2(u[1])])
    Omega = 1j*(kp*U[1] - kq*U[0])
    dt = cfl_max / (np.amax(u[0])/dx + np.amax(u[1])/dy)

    return Omega, dt

def init_P():
    P = np.zeros(shape = (N, 4, 2))
    P[:, 0, :] = rng.random((N, 2)) * 2 * np.pi
    
    # assuming particles are just dropped into flow, with no initial velocity, but will have fluid velocity + acceleration equal to that of the surrounding flow

    for n in range(0, N):
        P[n, 2, :] = np.array([
            + np.cos(beta*P[n, 0, 0]) * np.sin(beta*P[n, 0, 1]),
            - np.sin(beta*P[n, 0, 0]) * np.cos(beta*P[n, 0, 1])
        ])
        
        P[n, 3, :] = -nu * P[n, 2, :]
    
    return P

@numba.njit()
def calc_inter2D(U, x, y):
    # implementation of 2D Spectral Interpolation
    
    M = U.shape[0]

    u_inter = 0
    for p in numba.prange(0, M):
        u_yinter = 0
        for q in range(0, M):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y)

        u_yinter /= M
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x)

    u_inter /= M
    return u_inter

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
    
    # Acceleration Material Derivative Calculation
    A3 = (U3-U0)/dt # time component
    A3 += (1j*kp*U3[0] + 1j*kq*U3[1]) * U3 # convective component

    u3 = np.zeros(shape = U3.shape)
    u3[0] = real(ifft2(U3[0]))
    u3[1] = real(ifft2(U3[1]))
    dt_next = cfl_max / (np.amax(u3[0])/dx + np.amax(u3[1])/dy)

    return Omega3, U3, A3, dt_next

@numba.njit(parallel = True) 
def update_P(P0, U3, A3, dt):
    # note that `numba` does not like tuple unpacking, otherwise `Pp[n, 0, 0], Pp[n, 0, 1]` would be rewritten `*Pp[n, 0]`

    # P0: [x0, v0, u0, a0]
    Pp = np.empty(shape = P0.shape, dtype = np.float64)
    Pc = np.empty(shape = P0.shape, dtype = np.float64)

    if tau != 0:
        # Predictor-Corrector Scheme for Inertial Particles described in "Pressure statistics of gas nuclei in homogeneous isotropic turbulence with an application to cavitation inception"
        for n in numba.prange(0, N): # will auto-parallelize each particle
            # inertial particles predictor substep
            Pp[n, 0] = P0[n, 0] + dt*P0[n, 1] 
            
            Pp[n, 2, 0] = real(calc_inter2D(U3[0], Pp[n, 0, 0], Pp[n, 0, 1]))
            Pp[n, 2, 1] = real(calc_inter2D(U3[1], Pp[n, 0, 0], Pp[n, 0, 1]))
            
            if tau != 0: Pp[n, 1] = (P0[n, 1] + dt/tau * (Pp[n, 2] + 3*tau*P0[n, 3]))/(1+dt/tau)
            
            # inertial particles corrector substep
            Pc[n, 0] = (P0[n, 0] + dt/2 * (P0[n, 1] + Pp[n, 1])) % (2 * np.pi)
            
            Pc[n, 2, 0] = real(calc_inter2D(U3[0], Pc[n, 0, 0], Pc[n, 0, 1]))
            Pc[n, 2, 1] = real(calc_inter2D(U3[1], Pc[n, 0, 0], Pc[n, 0, 1]))
            
            if tau != 0: Pc[n, 3, 0] = real(calc_inter2D(A3[0], Pc[n, 0, 0], Pc[n, 0, 1]))
            if tau != 0: Pc[n, 3, 1] = real(calc_inter2D(A3[1], Pc[n, 0, 0], Pc[n, 0, 1]))
            
            if tau != 0: Pc[n, 1] = (2*tau)/(2*tau + dt) * ((1 - dt/tau/2)*P0[n, 1] + dt/2 * (3*(P0[n, 3] + Pc[n, 3]) + (P0[n, 2] + Pc[n, 2])/tau))

        return Pc
    
    if tau == 0:
        # Direct Heun's Method, updating particle position via surrounding fluid velocity, which the above method degenerates to for non-inertial particles, ie where `tau` is zero
        for n in numba.prange(0, N): # will auto-parallelize each particle
            # inertial particles predictor substep
            Pp[n, 0] = P0[n, 0] + dt*P0[n, 2]

            Pp[n, 2, 0] = real(calc_inter2D(U3[0], Pp[n, 0, 0], Pp[n, 0, 1]))
            Pp[n, 2, 1] = real(calc_inter2D(U3[1], Pp[n, 0, 0], Pp[n, 0, 1]))

            Pc[n, 2] = Pp[n, 2]
            Pc[n, 0] = P0[n, 0] + dt*(P0[n, 2] + Pp[n, 2])/2
        
        return Pc

@numba.njit()
def calc_tke(Omega):
    U = Omega * kU
    tke = np.sum(np.abs(U[0]) ** 2 + np.abs(U[1]) ** 2) # via Parseval's Theorem

    return tke

def save_animation(M, N, tau, s_max, s_int, savepath):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    eo = M // 32
    x_grid, y_grid = init_grid(M)[2:4]
    kU0, kU1 = init_wavearrays(M)[2:4]

    def calc_u(Omega):
        u = np.array([
            real(ifft2(kU0*Omega)), real(ifft2(kU1*Omega))
        ])
        u_mag = np.sqrt(u[0]**2 + u[1]**2)

        return u, u_mag
    
    data = np.load("data/" + savepath + "/v9 data s = 0.npz")
    Omega = data['Omega']
    P = data['P']
    u, u_mag = calc_u(Omega)

    fig, ax = plt.subplots(1, 1, constrained_layout = False)
    u_mag_plot = ax.pcolormesh(x_grid, y_grid, u_mag, cmap = 'coolwarm')
    u_vec_plot = ax.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo], u[0, ::eo, ::eo], u[1, ::eo, ::eo])
    part_plot = plt.scatter(P[:, 0, 0], P[:, 0, 1], color = 'black')

    ax.set_title(r"$\vec{u}(t$" + " = 0)")
    ax.set_aspect('equal')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax.yaxis.set_major_locator(MultipleLocator(base = np.pi))

    divider = make_axes_locatable(u_mag_plot.axes)
    cax = divider.append_axes("right", size = "5%", pad = 0.2)
    fig.colorbar(u_mag_plot, cax = cax, orientation = 'vertical')
    cax.set_ylabel(r"$\mathrm{mag}\left[ \vec{u} \right]$")

    def update_frame(s):
        data = np.load("data/" + savepath + "/v9 data s = {0}.npz".format(s))
        t = data['t']
        Omega = data['Omega']
        P = data['P']
        u, u_mag = calc_u(Omega)

        u_mag_plot.set_array(u_mag)
        u_vec_plot.set_UVC(u[0, ::eo, ::eo], u[1, ::eo, ::eo])
        part_plot.set_offsets(np.stack((P[:, 0, 0], P[:, 0, 1]), axis = 1))
        u_mag_plot.set_norm(Normalize(vmin = u_mag.min(), vmax = u_mag.max()))
        ax.set_title(r"$\vec{u}(t = $" + "{0:.5f})".format(t))

    ANI = FuncAnimation(fig, update_frame, init_func = lambda: None, frames = range(0, s_max, s_int)) # type: ignore
    ANI.save("anim/Inertial Particles v9 {0} {1} {2}.mp4".format(M, N, tau), fps = 180, dpi = 200, writer = 'ffmpeg')

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
    M = 128 # square computational grid dimensions
    N = 50 # number of particles
    T = 150 # end time
    nu = 2e-3 # kinematic viscosity, see note above
    cfl_max = 0.7 # Courant–Friedrichs–Lewy Condition Number
    tau = 1 # inertial time
    beta = 1 # periods within the [0, 2π) domain
    gamma = 0.02 # perturbation magnitude
    forced = True
    rng_seed = 123

    # initialization
    rng = np.random.default_rng(seed = rng_seed)
    x, y, x_grid, y_grid, dx, dy = init_grid(M)
    kp, kq, kU0, kU1, kU, kXi = init_wavearrays(M)
    Omega, dt = init_Omega()
    P = init_P()
    savepath = "{0} {1} {2} {3}".format(M, N, tau, forced)
    if not os.path.exists("data/" + savepath): os.mkdir("data/" + savepath)

    t = 0 # time
    s = 0 # step
    s_int = 5 # step/save interval
    while t < T:
        if s % s_int == 0:
            tke = calc_tke(Omega)
            print("n = {0:6}, t = {1:.5f}, tke = {2:.5e}".format(s, t, tke))
            np.savez("data/" + savepath + "/v9 data s = {0}.npz".format(s), t = t, Omega = Omega, P = P)  # type: ignore

        Omega, U, A, dt_next = update_Omega(Omega, dt)
        P = update_P(P, U, A, dt)
        
        t += dt
        s += 1

        dt = dt_next

    s_max = s

    save_animation(M, N, tau, s_max, s_int, savepath)