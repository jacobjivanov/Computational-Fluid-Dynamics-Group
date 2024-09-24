import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as ani
from time import time
from numpy.random import rand

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def RK3_step(pypt, y0, dt, *args): # version 11/03/2023
    """
    Performs the specialized 3rd-Order Runga Kutta method defined in "Spectral Methods for the Navier-Stokes Equations..." by Spalart, Moser, Rogers. Is able to process inputs of arbitrary dimensions, including scalar, vector, and larger arrays by flattening to a vector and later rebuilding the original dimensions. This function was last updated 10/28/2023 by JJI. 
    
    Function Inputs:
        pypt: time derivative of state y0 of form pypt(y0, *args). 
              must be autonomous (time-independent)
        y0: current state
        dt: time step
        *args: other pypt function arguments
    """

    y0 = np.asarray(y0)

    if y0.ndim == 0: # if scalar
        
        pypt_0 = pypt(y0)
        y_step1 = y0 + dt * 8/15 * pypt_0

        pypt_step1 = pypt(y_step1)
        y_step2 = y_step1 + dt * (-17/60 * pypt_0 + 5/12 * pypt_step1)

        pypt_step2 = pypt(y_step2)
        y_step3 = y_step2 + dt * (-5/12 * pypt_step1 + 3/4 * pypt_step2)
        
        return y_step3
    
    else: # if vector/array
        orig_shape = y0.shape
        orig_dtype = y0.dtype

        y0 = y0.flatten()
        N = y0.size

        pypt_0 = pypt(y0.reshape(orig_shape)).flatten()
        y_step1 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step1[i] = y0[i] + dt * 8/15 * pypt_0[i]

        pypt_step1 = pypt(y_step1.reshape(orig_shape)).flatten()
        y_step2 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step2[i] = y_step1[i] + dt * (-17/60 * pypt_0[i] + 5/12 * pypt_step1[i])

        pypt_step2 = pypt(y_step2.reshape(orig_shape)).flatten()
        y_step3 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step3[i] = y_step2[i] + dt * (-5/12 * pypt_step1[i] + 3/4 * pypt_step2[i])
    
        return y_step3.reshape(orig_shape)

x_dom, y_dom, t_end = [0, 2 * pi], [0, 2 * pi], 10
L_x, L_y = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0]
M, N = 64, 64
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N

nu = 5e-4
ar = 0.01

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
t = np.linspace(t_dom[0], t_dom[1], T, endpoint = False)
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

omega = np.empty(shape = (T, M, N), dtype = 'complex') # vorticity
psi = np.empty(shape = (T, M, N), dtype = 'complex') # streamfunction
u = np.empty(shape = (T, M, N), dtype = 'complex') # velocity (x-component
v = np.empty(shape = (T, M, N), dtype = 'complex') # velocity (y-component)

for i in range(0, M):
    for j in range(0, N):
        u[0, i, j] = + np.cos(x[i]) * np.sin(y[j]) + ar * (rand() - 0.5)
        v[0, i, j] = - np.sin(x[i]) * np.cos(y[j]) + ar * (rand() - 0.5)
U = np.empty(shape = (T, M, N), dtype = 'complex')
V = np.empty(shape = (T, M, N), dtype = 'complex')
U[0, :, :] = np.fft.fft2(u[0, :, :])
V[0, :, :] = np.fft.fft2(v[0, :, :])

Omega = np.empty(shape = (T, M, N), dtype = 'complex')
for p in range(0, M):
    for q in range(0, N):
        Omega[0, p, q] = 1j * (kx[p] * V[0, p, q] - ky[q] * U[0, p, q])
omega[0, :, :] = np.fft.ifft2(Omega[0, :, :])

k2 = np.empty(shape = (M, N))
for p in range(0, M):
    for q in range(0, N):
        # k2[p, q] = (-kx[p]**2 - ky[q]**2) if [p, q] != [0, 0] else np.inf
        k2[p, q] = (-kx[p]**2 - ky[q]**2)

def vorticity_timestep(Omega_curr):
    Psi_curr = np.empty(shape = (M, N), dtype = 'complex')
    U_curr = np.empty(shape = (M, N), dtype = 'complex')
    V_curr = np.empty(shape = (M, N), dtype = 'complex')
    Omega_x_curr = np.empty(shape = (M, N), dtype = 'complex')
    Omega_y_curr = np.empty(shape = (M, N), dtype = 'complex')
    lapOmega_curr = np.empty(shape = (M, N), dtype = 'complex')
    pOmegapt_curr = np.empty(shape = (M, N), dtype = 'complex')
    
    # Poisson Equation
    for p in range(0, M):
        for q in range(0, N):
            Psi_curr[p, q] = - Omega_curr[p, q] / k2[p, q]
    Psi_curr[0, 0] = 0

    # Streamfunction to Velocity Components
    for p in range(0, M):
        for q in range(0, N):
            U_curr[p, q] = + 1j * ky[q] * Psi_curr[p, q]
            V_curr[p, q] = - 1j * kx[p] * Psi_curr[p, q]
    global u_curr
    global v_curr
    u_curr = np.fft.ifft2(U_curr)
    v_curr = np.fft.ifft2(V_curr)

    # Laplace Omega Component
    for p in range(0, M):
        for q in range(0, N):
            lapOmega_curr[p, q] = k2[p, q] * Omega_curr[p, q]
    
    for p in range(0, M):
        for q in range(0, N):
            Omega_x_curr[p, q] = 1j * kx[p] * Omega_curr[p, q]
            Omega_y_curr[p, q] = 1j * ky[q] * Omega_curr[p, q]
    omega_x_curr = np.fft.ifft2(Omega_x_curr)
    omega_y_curr = np.fft.ifft2(Omega_y_curr)

    nonlin_1 = np.empty(shape = (M, N), dtype = 'complex')
    nonlin_2 = np.empty(shape = (M, N), dtype = 'complex')
    for i in range(0, M):
        for j in range(0, N):
            nonlin_1[i, j] = u_curr[i, j] * omega_x_curr[i, j]
            nonlin_2[i, j] = v_curr[i, j] * omega_y_curr[i, j]
    Nonlin_1 = np.fft.fft2(nonlin_1)
    Nonlin_2 = np.fft.fft2(nonlin_2)

    for p in range(0, M):
        for q in range(0, N):
            pOmegapt_curr[p, q] = nu * lapOmega_curr[p, q] - Nonlin_1[p, q] - Nonlin_2[p, q]

    global uv_curr
    uv_curr = np.sqrt(u_curr * u_curr + v_curr * v_curr)

    return pOmegapt_curr, Psi_curr, u_curr, v_curr

t_init = time()
for n in range(0, T - 1):
    CFL = np.amax(u_curr) / dx + np.amax(v_curr) / dy
    dt = 1 / CFL
    Omega[n + 1, :, :] = RK3_step(
        lambda Omega_curr: vorticity_timestep(Omega_curr)[0],
        Omega[n, :, :], 
        dt)
    
    # print("Vorticity in Fourier Space \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Vorticity in Fourier Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

t_init = time()
for n in range(1, T):
    omega[n, :, :] = np.fft.ifft2(Omega[n, :, :])
    print("Vorticity in Physical Space. \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Vorticity in Physical Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))