import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as ani
# import ffti_v10 as fi
from time import time

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def RK3_step(pypt, y_curr, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        y_t: current array of (∂y/∂t)[i, j]  
        y0: current array of y[i, j]
        dt: time step size
    """

    M, N = y_curr.shape

    y_step1 = np.empty(shape = (M, N))
    y_step2 = np.empty(shape = (M, N))
    y_step3  = np.empty(shape = (M, N))

    for i in range(0, M):
        for j in range(0, N):
            y_step3[i, j] = 0
            # currently not functional

    return y_step3

def RK1_step(pypt, y_curr, dt):
    """
    RK1_step does the Runga-Kutta 1 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)[i, j]  
        y0: current array of y[i, j]
        dt: time step size
    """
    M, N = y_curr.shape
    y_step1 = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            y_step1[i, j] = y_curr[i, j] + dt * pypt[i, j]
    return y_step1

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 10]
L_x, L_y, L_t = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0], t_dom[1]
M, N, T = 64, 64, 1000
dx, dy, dt = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N, t_dom[1]/T

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
t = np.linspace(t_dom[0], t_dom[1], T, endpoint = False)
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

u = np.empty(shape = (T, M, N))
U = np.empty(shape = (T, M, N), dtype = 'complex')

u_x = np.empty(shape = (T, M, N))
u_y = np.empty(shape = (T, M, N))
U_x = np.empty(shape = (T, M, N), dtype = 'complex')
U_y = np.empty(shape = (T, M, N), dtype = 'complex')

u_t = np.empty(shape = (T, M, N))

for i in range(0, M):
    for j in range(0, N):
        u[0, i, j] = np.exp(np.sin(x[i]) + np.cos(y[j]))

t_init = time()
for n in range(0, T - 1):
    U[n, :, :] = np.fft.fft2(u[n, :, :])

    for i in range(0, M):
        for j in range(0, N):
            U_x[n, i, j] = 1j * kx[i] * U[n, i, j]
            U_y[n, i, j] = 1j * ky[j] * U[n, i, j]
    u_x[n, :, :] = np.real(np.fft.ifft2(U_x[n, :, :]))
    u_y[n, :, :] = np.real(np.fft.ifft2(U_y[n, :, :]))

    for i in range(0, M):
        for j in range(0, N):
            u_t[n, i, j] = -2 * u_x[n, i, j] - 3 * u_y[n, i, j]

    u[n + 1, :, :] = RK1_step(u_t[n, :, :], u[n, :, :], dt)
    print("Solving Advection Equation. \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Solving Advection Equation. \tProgress: {0:07.3f}% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

# Figure Settings Below
fig = plt.figure()
ax = fig.add_subplot()

def init():
    u_plot = ax.pcolor(x, y, np.transpose(u[0, :, :]), cmap = 'coolwarm')
    fig.colorbar(u_plot)

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

    u_plot = ax.pcolor(x, y, np.transpose(u[frame, :, :]), cmap = 'coolwarm')
    ax.set_aspect('equal')
    ax.set_title(r"$u(t, x, y), t = $" + "{0:.3f}".format(t[frame]))
    # fig.colorbar(u_plot)

ANI = ani.FuncAnimation(fig, update, frames = range(0, T), init_func = init)
ANI.save("advection_animation_RK1.mp4", fps = 180, dpi = 200)