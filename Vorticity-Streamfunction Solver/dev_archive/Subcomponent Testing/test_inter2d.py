import numpy as np

# def inter_2D(kp, kq, U, pos):
#     M, N = U.shape

#     u_inter = 0
#     for p in range(0, M):
#         u_yinter = 0
#         for q in range(0, N):
#             u_yinter += U[p, q] * np.exp(1j * kq[q] * pos[1])

#         u_yinter /= N
#         u_inter += u_yinter * np.exp(1j * kp[p] * pos[0])

#     u_inter /= M
#     return u_inter

def inter_2D(kx, ky, U, pos):
    M, N = U.shape

    u_inter = 0
    for i in range(0, M):
        u_yinter = 0
        for j in range(0, N):
            u_yinter += U[i, j] * np.exp(1j * ky[i, j] * pos[1])

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kx[i, j] * pos[0])

    u_inter /= M
    return u_inter

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

x = np.linspace(0, 2 * np.pi, 100)
y = np.linspace(0, 2 * np.pi, 100)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')

u = np.exp(np.cos(x_grid) * np.sin(y_grid))
U = np.fft.fft2(u)

import matplotlib.pyplot as plt
plt.pcolormesh(u)
plt.show()
k_p, k_q = wavearrays(100, 100)

print(np.real(inter_2D(k_p, k_q, U, [1, 1])))