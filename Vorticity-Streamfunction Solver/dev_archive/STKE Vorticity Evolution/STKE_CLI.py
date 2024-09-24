# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# The purpose of this program is to provide a Command-Line Interface (CLI) for the STKE v1.1 Scalar Transport code, in order to help diagnose the numerical blowup problem

def STKE_CLI(M, nu, run_name, rng_seed):
    import numpy as np
    from numpy import real, imag
    from numpy.fft import fft2, ifft2
    from numpy.random import rand

    import numba as nb
    import matplotlib.pyplot as plt

    from STKE_helpers import wavearrays
    rng = np.random.default_rng(seed = rng_seed)

    # runtime dependent parameters
    N = M # computational grid dimensions
    Sc = 0.7 # schmidt number
    b = 1 # scalar transport gradient
    beta = 1 # periods within the [0, 2π) domain
    gamma = 0.02 # perturbation magnitude
    t_end = 200
    cfl_max = 0.8

    # boilerplate initialization
    x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
    dx, dy = x[1], y[1]
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    k_p, k_q, k_U, k_V = wavearrays(M, N)

    k_Theta_p = nu / Sc * (k_p**2 + k_q**2)
    k_Omega_u = nu * (k_p**2 + k_q**2)
    k_Omega_f = (k_p**2 + k_q**2)

    # forced turbulence viscosity modification
    for p in range(0, M):
        for q in range(0, N):
            if 5**2 <= k_Omega_f[p, q] and k_Omega_f[p, q] <= 6**2: k_Omega_f[p, q] *= -1
            if k_Omega_f[p, q] <= 2**2: k_Omega_f[p, q] *= 8
    k_Omega_f = nu * k_Omega_f

    x_pert = 2 * gamma * (rng.random((M, N)) - 0.5) # initial x-velocity perturbation
    y_pert = 2 * gamma * (rng.random((M, N)) - 0.5) # initial x-velocity perturbation
    u = +np.cos(beta*x_grid) * np.sin(beta*y_grid) + x_pert # physcial velocity x-component
    v = -np.sin(beta*x_grid) * np.cos(beta*y_grid) + y_pert # physical velocity y-component

    U, V = fft2(u), fft2(v)
    Omega = 1j*(k_p*V - k_q*U)
    omega = real(ifft2(Omega))

    # main loop functions
    @nb.njit()
    def Xi_Omega(t, TKE):
        if TKE < TKE0:
            return np.exp(k_Omega_f * t)
        else:
            return np.exp(k_Omega_u * t)

    @nb.njit()
    def Xi_Theta_p(t):
        return np.exp(k_Theta_p * t)

    @nb.njit()
    def pXiSTATEpt(t, XiOmega, XiTheta, TKE):
        Omega = Xi_Omega(-t, TKE) * XiOmega
        omega = ifft2(Omega)

        U, V = k_U * Omega, k_V * Omega
        u, v = real(ifft2(U)), real(ifft2(V))

        pXiOmegapt = -1j * Xi_Omega(t, TKE) * (k_p * fft2(u * omega) + k_q * fft2(v * omega))
        dt = cfl_max / (np.amax(u)/dx + np.amax(v)/dy)

        Theta_p = Xi_Theta_p(-t) * XiTheta
        Theta_p_x = 1j * k_p * Theta_p
        Theta_p_y = 1j * k_q * Theta_p
        theta_p_x, theta_p_y = real(ifft2(Theta_p_x)), real(ifft2(Theta_p_y))

        pXiTheta_ppt = - Xi_Theta_p(t) * (fft2(u * theta_p_x + v * theta_p_y) + b * V)

        return dt, pXiOmegapt, pXiTheta_ppt

    @nb.njit()
    def STATE_RK3step(dt, Omega, Theta_p, TKE):
        dt_next, k1O, k1T = pXiSTATEpt(
            0, 
            Omega,
            Theta_p,
            TKE)
        dt_next, k2O, k2T = pXiSTATEpt(
            dt * 8/15,
            Omega + dt * 8/15 * k1O,
            Theta_p + dt * 8/15 * k1T,
            TKE)
        dt_next, k3O, k3T = pXiSTATEpt(
            dt * 2/3,
            Omega + dt * (1/4 * k1O + 5/12 * k2O),
            Theta_p + dt * (1/4 * k1T + 5/12 * k2T),
            TKE)

        Omega += dt * (1/4 * k1O + 3/4 * k3O)
        Omega *= Xi_Omega(-dt, TKE)
        Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0

        Theta_p += dt * (1/4 * k1T + 3/4 * k3T)
        Theta_p *= Xi_Theta_p(-dt)
        Theta_p[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0

        U, V = k_U * Omega, k_V * Omega
        TKE = np.sum(np.abs(U) ** 2 + np.abs(V) ** 2)
        return dt_next, Omega, Theta_p, TKE

    t, n = 0, 0
    TKE0 = np.sum(np.abs(U) ** 2 + np.abs(V) ** 2)
    TKE = TKE0

    Theta_p = np.zeros(shape = (M, N), dtype = 'complex')
    dt_next = pXiSTATEpt(0, Omega, Theta_p, TKE)[0]

    while t < t_end:
        dt = dt_next
        
        if n % 20 == 0:
            print("{0}, {1:.5f}, {2:.5f}, {3:.5e}".format(n, dt, t, TKE))
            np.savez("temp/STKE v1.1 {0} {1}/Omega+Theta_p+t n = {2}.npz".format(M, run_name, n), Omega = Omega, Theta_p = Theta_p, t = t)

        dt_next, Omega, Theta_p, TKE = STATE_RK3step(dt, Omega, Theta_p, TKE)

        t += dt
        n += 1