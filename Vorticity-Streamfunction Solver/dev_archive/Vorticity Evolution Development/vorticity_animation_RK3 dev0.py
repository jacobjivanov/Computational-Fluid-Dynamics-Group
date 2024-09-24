import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as ani
from time import time

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def RK3_step(pypt, y0, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)[i, j]  
        y0: current array of y[i, j]
        dt: time step size
    """

    M, N = y0.shape

    y_step1 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_0 = pypt(y0)
    for i in range(0, M):
        for j in range(0, N):
            y_step1[i, j] = y0[i, j] + dt * 8/15 * pypt_0[i, j]

    y_step2 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_step1 = pypt(y_step1)
    for i in range(0, M):
        for j in range(0, N):
            y_step2[i, j] = y_step1[i, j] + dt * (-17/60 * pypt_0[i, j] + 5/12 * pypt_step1[i, j])

    y_step3 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_step2 = pypt(y_step2)
    for i in range(0, M):
        for j in range(0, N):
            y_step3[i, j] = y_step2[i, j] + dt * (-5/12 * pypt_step1[i, j] + 3/4 * pypt_step2[i, j])
    
    return y_step3

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 100]
L_x, L_y, L_t = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0], t_dom[1]
M, N, T = 64, 64, 10000
dx, dy, dt = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N, t_dom[1]/T
nu = 5e-4

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
        u[0, i, j] = + np.cos(x[i]) * np.sin(y[j])
        v[0, i, j] = - np.sin(x[i]) * np.cos(y[j])
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

def pOmegapt(Omega_curr):
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

    return pOmegapt_curr

t_init = time()
for n in range(0, T - 1):
    Omega[n + 1, :, :] = RK3_step(pOmegapt, Omega[n, :, :], dt)
    print("Vorticity in Fourier Space \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Vorticity in Fourier Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

t_init = time()
for n in range(1, T):
    omega[n, :, :] = np.fft.ifft2(Omega[n, :, :])
    print("Vorticity in Physical Space. \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Vorticity in Physical Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

omega = np.real(omega)

# NOTE: Figure Settings Below
fig = plt.figure()
ax = fig.add_subplot()

def init():
    omega_plot = ax.pcolor(x, y, np.transpose(omega[0, :, :]), cmap = 'coolwarm')
    fig.colorbar(omega_plot)

    return ax

def update(frame):
    f = int(frame)
    plt.cla()

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_xlim(0, 2 * pi)
    ax.set_ylim(0, 2 * pi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(MultipleLocator(base = pi))
    ax.yaxis.set_major_locator(MultipleLocator(base = pi))

    u_plot = ax.pcolor(x, y, np.transpose(omega[frame, :, :]), cmap = 'coolwarm')
    ax.set_aspect('equal')
    ax.set_title(r"$\frac{\partial u}{\partial t} + 2 \frac{\partial u}{\partial x} + 3 \frac{\partial u}{\partial y} = 0, t = $" + "{0:.3f}".format(t[frame]))
    # fig.colorbar(u_plot)

ANI = ani.FuncAnimation(fig, update, frames = range(0, T), init_func = init)
ANI.save("vorticity_animation_RK3.mp4", fps = 180, dpi = 200)