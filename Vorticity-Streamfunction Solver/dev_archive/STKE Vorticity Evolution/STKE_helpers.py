import numpy as np
import numba as nb

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))


    with np.errstate(divide = 'ignore', invalid = 'ignore'): # [p, q] = [0, 0] singularity
        k_U = +1j*k_q / (k_p**2 + k_q**2)
        k_V = -1j*k_p / (k_p**2 + k_q**2)
        k_Xi = k_p**2 + k_q**2
    k_U[0, 0], k_V[0, 0] = 0, 0

    return k_p, k_q, k_U, k_V

