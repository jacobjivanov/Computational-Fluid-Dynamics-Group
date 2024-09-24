# vorticity_evolution_v2.py

import numpy as np
from numpy.random import rand
import numba as nb
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time
import rocket_fft
rocket_fft.numpy_like()

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

@nb.njit()
def inter_2D(kx, ky, U, pos):
    M, N = U.shape

    u_inter = 0
    for i in nb.prange(0, M):
        u_yinter = 0
        for j in nb.prange(0, N):
            u_yinter += U[i, j] * np.exp(1j * ky[j] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kx[i] * pos[0])

    u_inter /= M
    return u_inter

# computational grid parameters
M, N = 64, 64
x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
dx, dy = x[1] - x[0], y[1] - y[0]
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

k2 = np.empty(shape = (M, N))
for p in range(0, M):
    for q in range(0, N):
        k2[p, q] = -kx[p]**2 - ky[q]**2

# fluid parameters
nu = 6e-3 # kinematic viscosity
ar = 0.0 # magnitude of initial perturbations
sc = 0.7 # schidt number
b = 0.2 # scalar gradient

# initial conditions
u = np.empty(shape = (M, N)) # physcial velocity x-component
v = np.empty(shape = (M, N)) # physical velocity y-component
for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(x[i]) * np.sin(y[j]) + 2 * ar * (rand() - 0.5)
        v[i, j] = - np.sin(x[i]) * np.cos(y[j]) + 2 * ar * (rand() - 0.5)

U, V = np.fft.fft2(u), np.fft.fft2(v)

Omega = np.empty(shape = (M, N), dtype = 'complex')
for p in range(0, M):
    for q in range(0, N):
        Omega[p, q] = 1j * (kx[p] * V[p, q] - ky[q] * U[p, q])
Omega[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

phi = np.empty(shape = (M, N), dtype = 'complex')
for i in range(0, M):
    for j in range(0, N):
        phi[i, j] = b * y[j]
Phi = np.fft.fft2(phi)

# computational time-step parameters
cfl_max = 0.50
t, t_end = 0, 200
n = 0

enable_particles = True # if False, `loc` is a dummy variable that stays constant
n_part = 1 # number of tracked particles
loc = np.empty(shape = (n_part, 2))
for p in range(0, n_part):
    loc[p, 0] = rand() * 2 * np.pi
    loc[p, 1] = rand() * 2 * np.pi

enable_figsaving = False
enable_animsaving = False

@nb.njit(parallel = True)
def RK3_step(pSTATEpt, Omega0, Phi0, loc0, dt): # version 12/03/2023
    """
    Performs the specialized 3rd-Order Runga Kutta method defined in "Spectral Methods for the Navier-Stokes Equations..." by Spalart, Moser, Rogers. Is only able to process 2D-arrays. This function was last updated 12/02/2023 by JJI. 
    
    Function Inputs:
        pypt: time derivative of state y0 of form pypt(Omega0, Phi0). 
              must be autonomous (time-independent)
        Omega0: current Omega state
        Phi0: current Phi state
        dt: time step
    """
     
    # RK Step 1
    pOmegapt_0, pPhipt_0, U_0, V_0, dt = pSTATEpt(Omega0, Phi0)
    Omega_step1 = np.empty(shape = (M, N), dtype = 'complex')
    Phi_step1 = np.empty(shape = (M, N), dtype = 'complex')
    for i in nb.prange(0, M):
        for j in nb.prange(0, N):
            Omega_step1[i, j] = Omega0[i, j] + dt * 8/15 * pOmegapt_0[i, j]
            Phi_step1[i, j] = Phi0[i, j] + dt * 8/15 * pPhipt_0[i, j]

    # RK Step 2
    pOmegapt_step1, pPhipt_step1, U_step1, V_step1, dt = pSTATEpt(Omega_step1, Phi_step1)
    Omega_step2 = np.empty(shape = (M, N), dtype = 'complex')
    Phi_step2 = np.empty(shape = (M, N), dtype = 'complex')
    for i in nb.prange(0, M):
        for j in nb.prange(0, N):
            Omega_step2[i, j] = Omega_step1[i, j] + dt * (-17/60 * pOmegapt_0[i, j] + 5/12 * pOmegapt_step1[i, j])
            Phi_step2[i, j] = Phi_step1[i, j] + dt * (-17/60 * pPhipt_0[i, j] + 5/12 * pPhipt_step1[i, j])

    # RK Step 3
    pOmegapt_step2, pPhipt_step2, U_step2, V_step2, dt = pSTATEpt(Omega_step2, Phi_step2)
    Omega_step3 = np.empty(shape = (M, N), dtype = 'complex')
    Phi_step3 = np.empty(shape = (M, N), dtype = 'complex')
    for i in nb.prange(0, M):
        for j in nb.prange(0, N):
            Omega_step3[i, j] = Omega_step2[i, j] + dt * (-5/12 * pOmegapt_step1[i, j] + 3/4 * pOmegapt_step2[i, j])
            Phi_step3[i, j] = Phi_step2[i, j] + dt * (-5/12 * pPhipt_step1[i, j] + 3/4 * pPhipt_step2[i, j])

    if enable_particles == True:
        loc_step1 = np.empty(shape = (n_part, 2))
        loc_step2 = np.empty(shape = (n_part, 2))
        loc_step3 = np.empty(shape = (n_part, 2))
        for p in nb.prange(0, n_part):
            loc_step1[p, 0] = loc0[p, 0] + dt * 8/15 * np.real(inter_2D(kx, ky, U_0, loc0[p]))
            loc_step1[p, 1] = loc0[p, 1] + dt * 8/15 * np.real(inter_2D(kx, ky, V_0, loc0[p]))

            loc_step2[p, 0] = loc_step1[p, 0] + dt * (-17/60 * np.real(inter_2D(kx, ky, U_0, loc0[p])) + 5/12 * np.real(inter_2D(kx, ky, U_step1, loc_step1[p])))
            loc_step2[p, 1] = loc_step1[p, 1] + dt * (-17/60 * np.real(inter_2D(kx, ky, V_0, loc0[p])) + 5/12 * np.real(inter_2D(kx, ky, V_step1, loc_step1[p])))

            loc_step3[p, 0] = loc_step2[p, 0] + dt * (-5/12 * np.real(inter_2D(kx, ky, U_step1, loc_step1[p]) + 3/4 * inter_2D(kx, ky, U_step2, loc_step2[p])))
            loc_step3[p, 1] = loc_step2[p, 1] + dt * (-5/12 * np.real(inter_2D(kx, ky, V_step1, loc_step1[p]) + 3/4 * inter_2D(kx, ky, V_step2, loc_step2[p])))

            loc_step3[p] = np.remainder(loc_step3[p], 2 * np.pi)
            
        return Omega_step3, Phi_step3, U_step2, V_step2, loc_step3, dt

    else:
        return Omega_step3, Phi_step3, U_step2, V_step2, loc0, dt

@nb.njit
def pSTATEpt(Omega, Phi):
    # recover frequency streamfunction
    Psi = np.empty(shape = (M, N), dtype = 'complex')
    Psi[0, 0] = 0
    for p in nb.prange(1, M):
        Psi[p, 0] = - Omega[p, 0] / k2[p, 0]
    for p in nb.prange(0, M):
        for q in nb.prange(1, N):
            Psi[p, q] = - Omega[p, q] / k2[p, q]

    # recover laplacian, x-, and y-derivatives of frequency vorticity
    lap_Omega = np.empty(shape = (M, N), dtype = 'complex') 
    Omega_x = np.empty(shape = (M, N), dtype = 'complex')
    Omega_y = np.empty(shape = (M, N), dtype = 'complex')
    for p in nb.prange(0, M):
        for q in nb.prange(0, N):
            lap_Omega[p, q] = k2[p, q] * Omega[p, q]
            Omega_x[p, q] = 1j * kx[p] * Omega[p, q]
            Omega_y[p, q] = 1j * ky[q] * Omega[p, q]
    
    # recover x-, and y-derivatives of physical vorticity
    omega_x, omega_y = np.fft.ifft2(Omega_x), np.fft.ifft2(Omega_y)

    # recover frequency x- and y-component velocities
    U = np.empty(shape = (M, N), dtype = 'complex')
    V = np.empty(shape = (M, N), dtype = 'complex')
    for p in nb.prange(0, M):
        for q in nb.prange(0, N):
            U[p, q] = + 1j * ky[q] * Psi[p, q]
            V[p, q] = - 1j * kx[p] * Psi[p, q]
    
    # recover physical x- and y-component velocities
    u, v = np.fft.ifft2(U), np.fft.ifft2(V)
    dt = cfl_max * (dx ** 2) / nu
    for i in nb.prange(0, M):
        for j in nb.prange(0, N):
            local_dt = cfl_max / (np.abs(u[i, j]) / dx + np.abs(v[i, j]) / dy)
            if local_dt < dt:
                dt = local_dt

    # recover mixed physical/frequency vorticity terms
    mixed_omega = u * omega_x + v * omega_y
    mixed_Omega = np.fft.fft2(mixed_omega)

    # recover final t-derivative of frequency vorticity
    pOmegapt = nu * lap_Omega - mixed_Omega
    pOmegapt[M//3 : 2 * M//3 + 1, N//3 : 2 * N//3 + 1] = 0 # high-frequency vorticity

    # recover laplacian, x-, and y-derivatives of frequency scalar transport
    lap_Phi = np.empty(shape = (M, N), dtype = 'complex')
    Phi_x = np.empty(shape = (M, N), dtype = 'complex')
    Phi_y = np.empty(shape = (M, N), dtype = 'complex')
    for p in nb.prange(0, M):
        for q in nb.prange(0, N):
            lap_Phi[p, q] = k2[p, q] * Phi[p, q]
            Phi_x[p, q] = 1j * kx[p] * Phi[p, q]
            Phi_y[p, q] = 1j * ky[q] * Phi[p, q]
    
    # recover mixed physical/frequency vorticity terms
    phi_x, phi_y = np.fft.ifft2(Phi_x), np.fft.ifft2(Phi_y)
    mixed_phi = u * phi_x + v * phi_y
    mixed_Phi = np.fft.fft2(mixed_phi)

    # recover t-derivative of scalar transport
    pPhipt = nu / sc * lap_Phi - mixed_Phi - b * V

    return pOmegapt, pPhipt, U, V, dt

def plot_routine(U, V, Phi, loc, t):
    u, v = np.real(np.fft.ifft2(U)), np.real(np.fft.ifft2(V))
        
    Dil = np.empty(shape = (M, N), dtype = "complex") # frequency dilatation
    for p in range(0, M):
        for q in range(0, N):
            Dil[p, q] =  1j * (kx[p] * U[p, q] + ky[q] * V[p, q])
    dil = np.real(np.fft.ifft2(Dil))
    
    phi = np.real(np.fft.ifft2(Phi))
    
    uv = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            uv[i, j] = (u[i, j] ** 2 + v[i, j] ** 2) ** (1/2)
    
    fig, ax = plt.subplots(1, 3, constrained_layout = True, figsize = (8, 5))
    fig.suptitle("\n\nVorticity-Streamfunction Evolution\n{0}% Perturbed Taylor-Greene Vortex Flow\nt = {1:.5f}, n = {2}".format(ar * 100, t, n))

    eo = M // 16
    ax[0].set_title(r"$\vec{u}$")
    vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
    if enable_particles == True:
        ax[0].scatter(loc.T[0], loc.T[1], marker = 'x', color = 'black')
    ax[0].quiver(x[::eo], y[::eo], u[::eo, ::eo].T, v[::eo, ::eo].T, pivot = 'mid')
    ax[0].set_aspect('equal')

    ax[1].set_title(r"$\nabla \cdot \vec{u}$")
    ax[1].set_aspect('equal')
    dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

    ax[2].set_title(r"$\phi$")
    ax[2].set_aspect('equal')
    phi_plot = ax[2].pcolormesh(x, y, np.transpose(phi), cmap = 'coolwarm')

    for plt_i in [0, 1, 2]:
        ax[plt_i].set_xlim(0, 2 * np.pi)
        ax[plt_i].set_ylim(0, 2 * np.pi)

        ax[plt_i].set_xlabel(r"$x$")
        ax[plt_i].set_ylabel(r"$y$")

        ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
        ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
        ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
        ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = np.pi))

    for plot in [vel_plot, dil_plot, phi_plot]:
        divider = make_axes_locatable(plot.axes)
        cax = divider.append_axes("bottom", size = "5%", pad = 0.5)
        fig.colorbar(plot, cax = cax, orientation = 'horizontal')
    
    return ax

dt = 0
interval = 100 # this is the data interval for plotting/animation/etc.
while t < t_end:
    if n % interval == 0: # plot procedure within
        if enable_animsaving == True:
            np.save("temp/U n = {}.npy".format(n), U)
            np.save("temp/V n = {}.npy".format(n), V)
            np.save("temp/Phi n = {}.npy".format(n), Phi)
            np.save("temp/loc n = {}.npy".format(n), loc)
            np.save("temp/t n = {}.npy".format(n), t)
            
            n_max = n # used later
        
        elif enable_figsaving == True:
                plot_routine(U, V, Phi, loc, t)
                plt.savefig("temp/figure n = {}.png".format(n), dpi = 200)
                plt.clf()
                plt.close()
        
        else:
            plot_routine(U, V, Phi, loc, t)                
            plt.show()

    # it should be noted that the `U` and `V` below are the 2/3 step within RK3. Since these variables are only used for plotting, where it is accurate enough, and as a dummy variable for intermediate steps, this is fine
    Omega, Phi, U, V, loc, dt = RK3_step(pSTATEpt, Omega, Phi, loc, dt)
    
    n += 1
    if n % 1 == 0:
        print("t = {0:.5f}, dt = {1:.5f} n = {2}".format(t, dt, n))
    
    t += dt

# animation proceduce below
if enable_animsaving == True:
    fig, ax = plt.subplots(1, 3, constrained_layout = True, figsize = (8, 5))

    def init():
        U = np.load("temp/U n = 0.npy")
        V = np.load("temp/V n = 0.npy")
        Phi = np.load("temp/Phi n = 0.npy")
        loc = np.load("temp/loc n = 0.npy")
        t = np.load("temp/t n = 0.npy")
        
        u, v = np.real(np.fft.ifft2(U)), np.real(np.fft.ifft2(V))
            
        Dil = np.empty(shape = (M, N), dtype = "complex") # frequency dilatation
        for p in range(0, M):
            for q in range(0, N):
                Dil[p, q] =  1j * (kx[p] * U[p, q] + ky[q] * V[p, q])
        dil = np.real(np.fft.ifft2(Dil))
        
        phi = np.real(np.fft.ifft2(Phi))
        
        uv = np.empty(shape = (M, N))
        for i in range(0, M):
            for j in range(0, N):
                uv[i, j] = (u[i, j] ** 2 + v[i, j] ** 2) ** (1/2)
        
        fig.suptitle("\n\nVorticity-Streamfunction Evolution\n{0}% Perturbed Taylor-Greene Vortex Flow\nt = {1:.5f}, n = {2}".format(ar * 100, t, n))

        eo = M // 16
        ax[0].set_title(r"$\vec{u}$")
        vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
        if enable_particles == True:
            ax[0].scatter(loc.T[0], loc.T[1], marker = 'x', color = 'black')
        ax[0].quiver(x[::eo], y[::eo], u[::eo, ::eo].T, v[::eo, ::eo].T, pivot = 'mid')
        ax[0].set_aspect('equal')

        ax[1].set_title(r"$\nabla \cdot \vec{u}$")
        ax[1].set_aspect('equal')
        dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

        ax[2].set_title(r"$\phi$")
        ax[2].set_aspect('equal')
        phi_plot = ax[2].pcolormesh(x, y, np.transpose(phi), cmap = 'coolwarm')

        for plt_i in [0, 1, 2]:
            ax[plt_i].set_xlim(0, 2 * np.pi)
            ax[plt_i].set_ylim(0, 2 * np.pi)

            ax[plt_i].set_xlabel(r"$x$")
            ax[plt_i].set_ylabel(r"$y$")

            ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
            ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = np.pi))

        for plot in [vel_plot, dil_plot, phi_plot]:
            divider = make_axes_locatable(plot.axes)
            cax = divider.append_axes("bottom", size = "5%", pad = 0.5)
            fig.colorbar(plot, cax = cax, orientation = 'horizontal')
        
        return ax

    def update(n):
        print("Animation {:10.5f}% complete".format(100 * n / n_max))
        plt.close()
        # plt.clf()
        
        U = np.load("temp/U n = {}.npy".format(n))
        V = np.load("temp/V n = {}.npy".format(n))
        Phi = np.load("temp/Phi n = {}.npy".format(n))
        loc = np.load("temp/loc n = {}.npy".format(n))
        t = np.load("temp/t n = {}.npy".format(n))
        
        u, v = np.real(np.fft.ifft2(U)), np.real(np.fft.ifft2(V))
            
        Dil = np.empty(shape = (M, N), dtype = "complex") # frequency dilatation
        for p in range(0, M):
            for q in range(0, N):
                Dil[p, q] =  1j * (kx[p] * U[p, q] + ky[q] * V[p, q])
        dil = np.real(np.fft.ifft2(Dil))
        
        phi = np.real(np.fft.ifft2(Phi))
        
        uv = np.empty(shape = (M, N))
        for i in range(0, M):
            for j in range(0, N):
                uv[i, j] = (u[i, j] ** 2 + v[i, j] ** 2) ** (1/2)
        
        fig.suptitle("\n\nVorticity-Streamfunction Evolution\n{0}% Perturbed Taylor-Greene Vortex Flow\nt = {1:.5f}, n = {2}".format(ar * 100, t, n))

        eo = M // 16
        ax[0].set_title(r"$\vec{u}$")
        vel_plot = ax[0].pcolormesh(x, y, np.transpose(uv), cmap = 'coolwarm')
        if enable_particles == True:
            ax[0].scatter(loc.T[0], loc.T[1], marker = 'x', color = 'black')
        ax[0].quiver(x[::eo], y[::eo], u[::eo, ::eo].T, v[::eo, ::eo].T, pivot = 'mid')
        ax[0].set_aspect('equal')

        ax[1].set_title(r"$\nabla \cdot \vec{u}$")
        ax[1].set_aspect('equal')
        dil_plot = ax[1].pcolormesh(x, y, np.transpose(dil), cmap = 'coolwarm')

        ax[2].set_title(r"$\phi$")
        ax[2].set_aspect('equal')
        phi_plot = ax[2].pcolormesh(x, y, np.transpose(phi), cmap = 'coolwarm')

        for plt_i in [0, 1, 2]:
            ax[plt_i].set_xlim(0, 2 * np.pi)
            ax[plt_i].set_ylim(0, 2 * np.pi)

            ax[plt_i].set_xlabel(r"$x$")
            ax[plt_i].set_ylabel(r"$y$")

            ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
            ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
            ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = np.pi))

        """
        for plot in [vel_plot, dil_plot, phi_plot]:
            divider = make_axes_locatable(plot.axes)
            cax = divider.append_axes("bottom", size = "5%", pad = 0.5)
            fig.colorbar(plot, cax = cax, orientation = 'horizontal')
        """

        return ax

    ANI = animation.FuncAnimation(fig, update, frames = np.arange(0, n_max + interval, interval), init_func = init)
    ANI.save("vorticity_evolution_v2.mp4", fps = 180, dpi = 200)