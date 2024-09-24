from mpi4py import MPI
import numpy as np
from numpy.fft import rfft2, irfft2, fft2, ifft2
import numba

def init_grid(M):
    x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, M, endpoint = False)
    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    dx, dy = x[1], y[1]

    return x, y, x_grid, y_grid, dx, dy

def init_wavearrays(M):
    kp = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1)).T

    kq = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (M, 1))

    return kp, kq

@numba.njit()
def calc_inter2D(kp, kq, U, x, y):
    M = U.shape[0]

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, M):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y)

        u_yinter /= M
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x)

    u_inter /= M
    return u_inter

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # runtime parameters
    M = 4
    rng_seed = 123
    rng = np.random.default_rng(seed = rng_seed)

    if rank == 0: # main node
        x, y, x_grid, y_grid, dx, dy = init_grid(M)

        u = np.exp(np.sin(x_grid) + np.cos(y_grid))
        U = fft2(u)

        U_flat = U.flatten()
        comm.Bcast(U_flat, root = 0)

        comm.

    if rank != 0: # satellite node

        kp, kq = init_wavearrays(M)

        U_flat = np.empty(M*M, dtype = 'complex')
        comm.Bcast(U_flat, root = 0)
        U = np.reshape(U_flat, (M, M))
        
        for i in range(0, 100000):
            x_int = 2*np.pi * rng.random()
            y_int = 2*np.pi * rng.random()
            u_int = calc_inter2D(kp, kq, U, x_int, y_int)
            print("(x, y, u) = ({0:.5f}, {1:.5f}, {2:.5f})".format(x_int, y_int, u_int))