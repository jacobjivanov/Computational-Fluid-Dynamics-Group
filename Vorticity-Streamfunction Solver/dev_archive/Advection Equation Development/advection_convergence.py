import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numba

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

@numba.njit()
def rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + 8/15*dt, y0 + dt*8/15*k1)
    k3 = f(t0 + 2/3*dt, y0 + dt * (k1/4 + 5*k2/12))

    y3 = y0 + dt * (k1/4 + 3*k3/4)
    return y3

def f(x, y): return np.exp(np.sin(x) + np.cos(y))
a, b = 2, 3
# cfl = 0.5

x_dom, y_dom, t_dom = [0, 2 * pi], [0, 2 * pi], [0, 100]
L_x, L_y, L_t = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0], t_dom[1]
M, N = 64, 64
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N
# dt = cfl / (a/dx + b/dy)

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

u_num = np.empty(shape = (M, N))
U_num = np.empty(shape = (M, N), dtype = 'complex')
u_ana = np.empty(shape = (M, N))

@numba.njit()
def pUpt(t, U):
    M, N = U.shape

    U_x = np.empty(shape = (M, N), dtype = 'complex')
    U_y = np.empty(shape = (M, N), dtype = 'complex')

    for i in range(0, M):
        for j in range(0, N):
            U_x[i, j] = 1j * kx[i] * U[i, j]
            U_y[i, j] = 1j * ky[j] * U[i, j]

    U_t = - a * U_x - b * U_y
    return U_t

T = np.array([1e0, 2e0, 3e0, 4e0, 5e0, 6e0, 7e0, 8e0, 9e0, 1e1, 2e1, 3e1, 4e1, 5e1, 6e1, 7e1, 8e1, 9e1, 1e2, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 2e4, 5e4])
t_end = 1

for i in range(0, M):
    for j in range(0, N):
        u_ana[i, j] = f(x[i] - a * t_end, y[j] - b * t_end)

del_t = t_end / T
l1_error = np.empty(len(T))
l2_error = np.empty(len(T))

for t in range(0, len(T)):
    print(T[t])
    for i in range(0, M):
        for j in range(0, N):
            u_num[i, j] = f(x[i], y[j])
    U_num = np.fft.fft2(u_num)

    for n in range(0, int(T[t])):
        U_num = rk3_step(pUpt, t, U_num, dt = del_t[t])
    u_num = np.real(np.fft.ifft2(U_num))

    e = u_num - u_ana
    
    """
    plt.pcolor(np.transpose(u_ana))
    plt.colorbar()
    plt.show()
    """

    l1_error[t] = lp_norm(e, dx, dy, 1)
    l2_error[t] = lp_norm(e, dx, dy, 2)

fig, ax = plt.subplots()
ax.set_title("Convergence of Advection Equation Solution\n" + r"$t \in [0, 1]$ as $\Delta t \to 0$")
ax.set_xlabel("$\Delta t$")
ax.set_ylabel(r"$ \ell^p \left[u_{\mathrm{num}} - u_{\mathrm{ana}} \right]$")
ax.loglog(del_t, l1_error, 'o', label = "p = 1", color = 'red')
ax.loglog(del_t, l2_error, 'o', label = "p = 2", color = 'blue')
ax.legend()

plt.show()
# plt.savefig("advection_convergence.png", dpi = 600)
