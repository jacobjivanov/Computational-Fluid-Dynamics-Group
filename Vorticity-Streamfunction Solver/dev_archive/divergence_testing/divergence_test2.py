import numpy as np
from numpy import pi, real
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.fft import fft2, ifft2
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        
        pypt_0 = pypt(y0, *args)
        y_step1 = y0 + dt * 8/15 * pypt_0

        pypt_step1 = pypt(y_step1, *args)
        y_step2 = y_step1 + dt * (-17/60 * pypt_0 + 5/12 * pypt_step1)

        pypt_step2 = pypt(y_step2, *args)
        y_step3 = y_step2 + dt * (-5/12 * pypt_step1 + 3/4 * pypt_step2)
        
        return y_step3
    
    else: # if vector/array
        orig_shape = y0.shape
        orig_dtype = y0.dtype

        y0 = y0.flatten()
        N = y0.size

        pypt_0 = pypt(y0.reshape(orig_shape), *args).flatten()
        y_step1 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step1[i] = y0[i] + dt * 8/15 * pypt_0[i]

        pypt_step1 = pypt(y_step1.reshape(orig_shape), *args).flatten()
        y_step2 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step2[i] = y_step1[i] + dt * (-17/60 * pypt_0[i] + 5/12 * pypt_step1[i])

        pypt_step2 = pypt(y_step2.reshape(orig_shape), *args).flatten()
        y_step3 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step3[i] = y_step2[i] + dt * (-5/12 * pypt_step1[i] + 3/4 * pypt_step2[i])
    
        return y_step3.reshape(orig_shape)

# computational grid parameters
M, N = 64, 64
x_dom, y_dom = [0, 2 * pi], [0, 2 * pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
dx, dy = x[1] - x[0], y[1] - y[0]
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

k2 = np.empty(shape = (M, N))
for p in range(0, M):
    for q in range(0, N):
        k2[p, q] = -kx[p]**2 - ky[q]**2

# computational time-step parameters
cfl_max = 0.10
t, t_end = 0, 10000
n = 0

# fluid parameters
nu = 6e-3 # kinematic viscosity
ar = 0.50 # magnitude of initial perturbations
sc = 0.7 # schidt number
b = 0.2 # scalar gradient

# initial conditions
alpha = 1 # vortex repetition (not added in yet), i.e. sin(alpha x)...

u = np.empty(shape = (M, N)) # physcial velocity x-component
v = np.empty(shape = (M, N)) # physical velocity y-component
psi = np.empty(shape = (M, N)) # physical streamfunction
omega = np.empty(shape = (M, N)) # physical vorticity

for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(x[i]) * np.sin(y[j]) + ar * (rand() - 0.5)
        v[i, j] = - np.sin(x[i]) * np.cos(y[j]) + ar * (rand() - 0.5)
U = fft2(u)
V = fft2(v)

Omega = np.empty(shape = (M, N), dtype = 'complex')
for p in range(0, M):
    for q in range(0, N):
        Omega[p, q] = 1j * (kx[p] * V[p, q] - ky[q] * U[p, q])
omega = np.real(ifft2(Omega))

def recover_Psi(Omega):
    Psi = np.empty(shape = (M, N), dtype = 'complex') # frequency streamfunction
    
    Psi[0, 0] = 0 # avoids the k2[0, 0] singularity
    for p in range(1, M):
        Psi[p, 0] = - Omega[p, 0] / k2[p, 0]

    for p in range(0, M):
        for q in range(1, N):
            Psi[p, q] = - Omega[p, q] / k2[p, q]

    return Psi

def recover_u(Psi):
    U = np.empty(shape = (M, N), dtype = 'complex') # frequency velocity x-component
    
    for p in range(0, M):
        for q in range(0, N):
            U[p, q] = + 1j * ky[q] * Psi[p, q]
    
    u = ifft2(U) # physical velocity x-component
    return u

def recover_v(Psi):
    V = np.empty(shape = (M, N), dtype = 'complex') # frequency velocity x-component
    
    for p in range(0, M):
        for q in range(0, N):
            V[p, q] = - 1j * kx[p] * Psi[p, q]
    
    v = ifft2(V) # physical velocity y-component
    return v

def recover_lap_Omega(Omega):
    lap_Omega = np.empty(shape = (M, N), dtype = 'complex') # laplacian of frequency vorticity (for diffusion term)

    for p in range(0, M):
        for q in range(0, N):
            lap_Omega[p, q] = k2[p, q] * Omega[p, q]

    return lap_Omega

def recover_detjac_PsiOmega(Omega, u, v):
    # Term stands for determinant of jacobian of Psi, Omega with respect to (y, x). Didn't know what else to call it.
    # \mathcal{F} \left| \mathbf{J}_{\psi, \omega} \right| = - \mathcal{F} \left[ u \cdot \omega_x + v \cdot \omega_y \right]

    Omega_x = np.empty(shape = (M, N), dtype = 'complex') # x-derivative of frequency vorticity
    Omega_y = np.empty(shape = (M, N), dtype = 'complex') # y-derivative of frequency vorticity
    detjac_psiomega = np.empty(shape = (M, N), dtype = 'complex') # determinant of jacobian of (\psi, \omega) with respect to (y, x)

    for p in range(0, M):
        for q in range(0, N):
            Omega_x[p, q] = 1j * kx[p] * Omega[p, q]
            Omega_y[p, q] = 1j * ky[q] * Omega[p, q]
    
    omega_x = ifft2(Omega_x) # x-derivative of physical vorticity
    omega_y = ifft2(Omega_y) # y-derivative of physical vorticity

    for i in range(0, M):
        for j in range(0, N):
            detjac_psiomega[i, j] = - (u[i, j] * omega_x[i, j] + v[i, j] * omega_y[i, j])
    
    detjac_PsiOmega = fft2(detjac_psiomega)
    return detjac_PsiOmega

def pOmegapt(Omega):
    Psi = recover_Psi(Omega)
    u, v = recover_u(Psi), recover_v(Psi)
    lap_Omega = recover_lap_Omega(Omega)
    detjac_PsiOmega = recover_detjac_PsiOmega(Omega, u, v)

    pOmegapt = np.empty(shape = (M, N), dtype = 'complex')
    for p in range(0, M):
        for q in range(0, N):
            pOmegapt[p, q] = nu * lap_Omega[p, q] + detjac_PsiOmega[p, q]
    
    pOmegapt[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

    return pOmegapt

while t < t_end:
    if n % 500 == 0: # plot procedure within
        Psi = recover_Psi(Omega)
        
        U = np.empty(shape = (M, N), dtype = 'complex')
        for p in range(0, M):
            for q in range(0, N):
                U[p, q] = + 1j * ky[q] * Psi[p, q]

        V = np.empty(shape = (M, N), dtype = 'complex')
        for p in range(0, M):
            for q in range(0, N):
                V[p, q] = - 1j * kx[p] * Psi[p, q]
        
        u, v = real(ifft2(U)), real(ifft2(V))
        uv = np.empty(shape = (M, N))
        for i in range(0, M):
            for j in range(0, N):
                uv[i, j] = (u[i, j] ** 2 + v[i, j] ** 2) ** (1/2)
        
        Dil = np.empty(shape = (M, N), dtype = "complex") # frequency dilatation
        for p in range(0, M):
            for q in range(0, N):
                Dil[p, q] =  1j * (kx[p] * U[p, q] + ky[q] * V[p, q])
        dil = np.real(ifft2(Dil))
        
        fig, ax = plt.subplots(1, 2, constrained_layout = True)
        fig.suptitle("Vorticity-Streamfunction Evolution\nTaylor-Greene Vortex Flow, Divergence Test\nt = {0:.5f}, n = {1}".format(t, n))

        eo = 4
        ax[0].set_title(r"$\vec{u}$")
        vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
        ax[0].quiver(x[: : eo], y[: : eo], u[: : eo, : : eo], v[: : eo, : : eo])
        ax[0].set_aspect('equal')

        ax[1].set_title(r"$\nabla \cdot \vec{u}$")
        ax[1].set_aspect('equal')
        dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

        for plt_i in [0, 1]:
            ax[plt_i].set_xlim(0, 2 * pi)
            ax[plt_i].set_ylim(0, 2 * pi)

            ax[plt_i].set_xlabel(r"$x$")
            ax[plt_i].set_ylabel(r"$y$")

            ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
            ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
            ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = pi))
            ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = pi))

        for plot in [vel_plot, dil_plot]:
            divider = make_axes_locatable(plot.axes)
            cax = divider.append_axes("bottom", size = "5%", pad = 0.3)
            fig.colorbar(plot, cax = cax, orientation = 'horizontal')

        plt.show()

    n += 1

    # calculates the maximum dt value based on velocity/viscosity values at every array location, then takes the minimum to maintain stability
    local_dt = np.empty(shape = (M, N))
    # vis_dt = cfl_max * (dx ** 2 + dy ** 2) / nu
    vis_dt = cfl_max * (dx ** 2) / nu
    for i in range(0, M):
        for j in range(0, N):
            adv_dt = cfl_max / (abs(u[i, j]) / dx + abs(v[i, j]) / dy)
            local_dt[i, j] = min(adv_dt, vis_dt) 
    dt = np.amin(local_dt)

    Omega = RK3_step(pOmegapt, Omega, dt)
    Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

    print("t = {0:.5f}, dt = {1:.5f} n = {2}".format(t, dt, n))
    t += dt