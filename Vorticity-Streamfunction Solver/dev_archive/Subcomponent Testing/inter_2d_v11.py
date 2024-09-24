import numpy as np
import numba as nb

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

@nb.njit
def inter_2D(kx, ky, U, pos):
    M, N = U.shape

    u_inter = 0
    for i in range(0, M):
        u_yinter = 0
        for j in range(0, N):
            u_yinter += U[i, j] * np.exp(1j * ky[j] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kx[i] * pos[0])

    u_inter /= M
    return u_inter

M, N = 64, 16
x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
dx, dy = x[1] - x[0], y[1] - y[0]
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

u = np.empty(shape = (M, N)) # physcial velocity x-component
for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(x[i]) * np.sin(y[j])

U = np.fft.fft2(u)

a = inter_2D(kx, ky, U, np.array([1, 2.3]))
print(a)