import numpy as np

M, N = 1235, 217
x_dom, y_dom = [0, 4 * np.pi], [0, 2 * np.pi]

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

k_p, k_q = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

def wavearrays(M, N, x_dom, y_dom):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))) * 2 * np.pi / (x_dom[1] - x_dom[0]), reps = (N, 1)
    ).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (y_dom[1] - y_dom[0]), reps = (M, 1)
    )

    return k_p, k_q

kp, kq = wavearrays(M, N, x_dom, y_dom)

print(k_q[12], kq[12, 12])