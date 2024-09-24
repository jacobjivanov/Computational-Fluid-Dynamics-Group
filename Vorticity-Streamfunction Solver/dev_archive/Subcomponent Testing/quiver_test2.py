import numpy as np
import matplotlib.pyplot as plt

M, N = 32, 32
x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')

u = np.sin(x_grid)
v = - np.cos(y_grid)

U, V = np.fft.fft2(u), np.fft.fft2(v)

n_p = 10
px = np.random.rand(n_p) * 2 * np.pi
py = np.random.rand(n_p) * 2 * np.pi

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

k_p, k_q = wavearrays(M, N)

def inter_2D(kp, kq, U, pos):
    M, N = U.shape

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, N):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * pos[0])

    u_inter /= M
    return u_inter

dt = 0.1
for i in range(0, 1000):
    for n in range(0, n_p):
        px[n] = (px[n] + dt * np.real(inter_2D(k_p, k_q, U, pos = [px[n], py[n]]))) % (2 * np.pi)
        py[n] = (py[n] + dt * np.real(inter_2D(k_p, k_q, V, pos = [px[n], py[n]]))) % (2 * np.pi)
    plt.quiver(x_grid, y_grid, u, v, pivot = 'mid')
    plt.scatter(px, py)
    plt.show()