import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import ffti_v10 as fi

"""
def fi.wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])
"""

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
        y_t: current array of (∂y/∂t)[i, j]  
        y0: current array of y[i, j]
        dt: time step size
    """

    M, N = y_curr.shape
    y_step1 = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            y_step1[i, j] = y_curr[i, j] + dt * pypt[i, j]
    return y_step1

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 100]
L_x, L_y = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0]
M, N = 256, 256
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
kx, ky = fi.wavenumbers(M, x_dom), fi.wavenumbers(N, y_dom)

u = np.empty(shape = (M, N))
U = np.empty(shape = (M, N), dtype = 'complex')
for i in range(0, M):
    for j in range(0, N):
        u[i, j] = np.exp(np.sin(x[i]) + np.cos(y[j]))
U = np.fft.fft2(u)

u_x = np.empty(shape = (M, N))
u_y = np.empty(shape = (M, N))
U_x = np.empty(shape = (M, N), dtype = 'complex')
U_y = np.empty(shape = (M, N), dtype = 'complex')
u_x_ana = np.empty(shape = (M, N))
u_y_ana = np.empty(shape = (M, N))
for i in range(0, M):
    for j in range(0, N):
        U_x[i, j] = 1j * kx[i] * U[i, j]
        U_y[i, j] = 1j * ky[j] * U[i, j]

        u_x_ana[i, j] = u[i, j] * np.cos(x[i])
        u_y_ana[i, j] = - u[i, j] * np.sin(y[j])
u_x = np.real(np.fft.ifft2(U_x))
u_y = np.real(np.fft.ifft2(U_y))

u_t = np.empty(shape = (M, N))
for i in range(0, M):
    for j in range(0, N):
        u_t[i, j] = -2 * u_x[i, j] - 3 * u_y[i, j]

# u = RK1_step(u_t, u, 0.01)

fig, ax = plt.subplots(2, 2, constrained_layout = True)
fig.suptitle("Advection Equation Terms\nComputational Grid Dimensions: {0}x{1}".format(M, N))

u_plot = ax[0, 0].pcolor(x, y, np.transpose(u), cmap = 'coolwarm')
ax[0, 0].set_aspect('equal')
ax[0, 0].set_title(r"Defined $u$")
fig.colorbar(u_plot, ax = ax[0, 0])

ux_plot = ax[0, 1].pcolor(x, y, np.transpose(u_x - u_x_ana), cmap = 'coolwarm')
ax[0, 1].set_aspect('equal')
ax[0, 1].set_title(r"$u_x$ error")
fig.colorbar(ux_plot, ax = ax[0, 1])

uy_plot = ax[1, 0].pcolor(x, y, np.transpose(u_y - u_y_ana), cmap = 'coolwarm')
ax[1, 0].set_aspect('equal')
ax[1, 0].set_title(r"$u_y$ error")
fig.colorbar(uy_plot, ax = ax[1, 0])

ut_plot = ax[1, 1].pcolor(x, y, np.transpose(u_t), cmap = 'coolwarm')
ax[1, 1].set_aspect('equal')
ax[1, 1].set_title(r"$u_t$")
fig.colorbar(ut_plot, ax = ax[1, 1])

for plt_i in [0, 1]:
    for plt_j in [0, 1]:
        ax[plt_i, plt_j].set_xlim(0, 2 * pi)
        ax[plt_i, plt_j].set_ylim(0, 2 * pi)

        ax[plt_i, plt_j].set_xlabel(r"$x$")
        ax[plt_i, plt_j].set_ylabel(r"$y$")

        ax[plt_i, plt_j].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
        ax[plt_i, plt_j].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
        ax[plt_i, plt_j].xaxis.set_major_locator(MultipleLocator(base = pi))
        ax[plt_i, plt_j].yaxis.set_major_locator(MultipleLocator(base = pi))

plt.show()
# fig.savefig("Advection Equation {0}x{1}.png".format(M, N), dpi = 400)
