import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def lp_norm(e, dx, dy, p):
    M, N = e.shape
    lp = 0
    for i in range(0, M):
        for j in range(0, N):
            lp += np.abs(e[i, j]) ** p
    lp = (dx * dy * lp) ** (1/p)
    return lp

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

def f(x, y): return np.exp(np.sin(x) + np.cos(y))
a, b = 2, 3
cfl = 0.5

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 100]
L_x, L_y, L_t = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0], t_dom[1]
M, N = 64, 64
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N
dt = cfl / (a/dx + b/dy)
T = int(np.ceil(t_dom[1] / dt))

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
t = np.linspace(t_dom[0], dt * T, T, endpoint = False)
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

u_num = np.empty(shape = (M, N))
U_num = np.empty(shape = (M, N), dtype = 'complex')
u_ana = np.empty(shape = (M, N))

for i in range(0, M):
    for j in range(0, N):
        u_num[i, j] = f(x[i], y[j])
U_num = np.fft.fft2(u_num)

def pUpt(U_curr):
    M, N = U_curr.shape

    U_x_curr = np.empty(shape = (M, N), dtype = 'complex')
    U_y_curr = np.empty(shape = (M, N), dtype = 'complex')

    for i in range(0, M):
        for j in range(0, N):
            U_x_curr[i, j] = 1j * kx[i] * U_curr[i, j]
            U_y_curr[i, j] = 1j * ky[j] * U_curr[i, j]

    U_t_curr = - a * U_x_curr - b * U_y_curr
    return U_t_curr

l1_error = np.empty(T)
l2_error = np.empty(T)
l1_error[0], l2_error[0] = 0, 0

tke_old = 0
for n in range(1, T):
    U_num = RK3_step(pUpt, U_num, dt)

    tke = 0.0
    for i in range(0, M):
        for j in range(0, N):
            tke += abs(U_num[i, j] ** 2)
    print("t = {0:.5f}, Δtke = {1:.5e}".format(t[n], tke - tke_old))
    tke_old = tke

    u_num = np.real(np.fft.ifft2(U_num))

    for i in range(0, M):
        for j in range(0, N):
            u_ana[i, j] = f(x[i] - a * t[n], y[j] - b * t[n])
    
    e = u_num - u_ana

    # Since matplotlib will auto-pause the program at every plt.show, you can see the error grow with every time-step by un-commenting the next three lines.
    """
    fig, ax = plt.subplots()
    u_plot = ax.pcolor(x, y, np.transpose(e), cmap = 'coolwarm')
    fig.colorbar(u_plot)
    ax.set_title(r"$u_{\mathrm{num}} - u_{\mathrm{ana}}$, t = " + "{0:.5f}".format(t[n]))

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_xlim(0, 2 * pi)
    ax.set_ylim(0, 2 * pi)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(MultipleLocator(base = pi))
    ax.yaxis.set_major_locator(MultipleLocator(base = pi))
    plt.show()
    """

    l1_error[n] = lp_norm(e, dx, dy, 1)
    l2_error[n] = lp_norm(e, dx, dy, 2)

plt.title(r"Normalized Error Growth for $\frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} + b \frac{\partial u}{\partial y} = 0$" + "\n" + r"$u(t = 0) = \exp \left[ \sin(x) \cos(y) \right]$, $a = 2$, $b = 3$, $t \in [0, 100]$, $CFL = 0.2$")
plt.loglog(t, l1_error, label = "p = 1")
plt.loglog(t, l2_error, label = "p = 2")
plt.ylabel(r"$ \ell^p \left[u_{\mathrm{num}} - u_{\mathrm{ana}} \right]$")
plt.xlabel("$t$")
plt.legend()
plt.show()