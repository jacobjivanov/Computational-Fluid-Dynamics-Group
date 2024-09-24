import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as ani
# import ffti_v10 as fi
from time import time

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def RK3v0_step(pypt, y0, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)[i, j]  Jj
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

def RK3_step(pypt, y0, dt, *args):
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
    
    else:
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

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 100]
L_x, L_y, L_t = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0], t_dom[1]
M, N, T = 32, 32, 4500
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
U_t = np.empty(shape = (T, M, N), dtype = 'complex')

for i in range(0, M):
    for j in range(0, N):
        u[0, i, j] = np.exp(np.sin(x[i]) + np.cos(y[j]))

U[0, :, :] = np.fft.fft2(u[0, :, :])

def pUpt(U_curr):
    M, N = U_curr.shape

    U_x_curr = np.empty(shape = (M, N), dtype = 'complex')
    U_y_curr = np.empty(shape = (M, N), dtype = 'complex')

    for i in range(0, M):
        for j in range(0, N):
            U_x_curr[i, j] = 1j * kx[i] * U_curr[i, j]
            U_y_curr[i, j] = 1j * ky[j] * U_curr[i, j]

    U_t_curr = -2 * U_x_curr - 3 * U_y_curr
    return U_t_curr

t_init = time()
tke_old = 0
for n in range(0, T - 1):
    U[n + 1, :, :] = RK3_step(pUpt, U[n, :, :], dt)

    tke = 0.0
    for i in range(0, M):
        for j in range(0, N):
            tke += abs(U[n + 1, i, j] ** 2)
    print(tke_old - tke)
    tke_old = tke

    # print("Advection in Fourier Space \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Advection in Fourier Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

t_init = time()
for n in range(1, T):
    u[n, :, :] = np.real(np.fft.ifft2(U[n, :, :]))
    print("Converting to Physical Space. \tProgress: {0:07.3f}% Complete".format(100 * n / T), end = '\r')
t_build = time()
print("Converting to Physical Space. \tProgress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))


# NOTE: Figure Settings Below
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
    ax.set_title(r"$\frac{\partial u}{\partial t} + 2 \frac{\partial u}{\partial x} + 3 \frac{\partial u}{\partial y} = 0, t = $" + "{0:.3f}".format(t[frame]))
    # fig.colorbar(u_plot)

ANI = ani.FuncAnimation(fig, update, frames = range(0, T), init_func = init)
ANI.save("advection_animation_RK3.mp4", fps = 180, dpi = 200)