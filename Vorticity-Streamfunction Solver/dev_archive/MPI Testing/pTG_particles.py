# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# The goal of this program is to implement non-inertial particle transport within unperturbed Taylor-Green flow. Additionally, it will be parallelized with `mpi4py`.

import numpy as np
from numpy.fft import fft2, ifft2
from mpi4py import MPI

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
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        kU0 = +1j*kq / (kp**2 + kq**2)
        kU1 = -1j*kp / (kp**2 + kq**2)
        kXi = kp**2 + kq**2
    kU0[0, 0], kU1[0, 0] = 0, 0
    kU = np.array([kU0, kU1])

    return kp, kq, kU

def init_Omega(beta):
    u = np.array([
        +np.cos(beta*x_grid) * np.sin(beta*y_grid),
        -np.sin(beta*x_grid) * np.cos(beta*y_grid),
    ])

    U = np.array([fft2(u[0]), fft2(u[1])])
    Omega = 1j*(kp*U[1] - kq*U[0])

    return Omega

def init_1P():
    thisP = np.zeros(shape = (1, 4, 2))
    thisP[0, 0, 0] =  2 * np.pi * rng.random()
    thisP[0, 0, 1] =  2 * np.pi * rng.random()
    
    thisP[0, 2, 0] = + np.cos(beta*thisP[0, 0, 0]) * np.sin(beta*thisP[0, 0, 1])
    thisP[0, 2, 1] = - np.sin(beta*thisP[0, 0, 0]) * np.cos(beta*thisP[0, 0, 1])

    thisP[0, 3] = -nu * thisP[0, 2]

def init_P():
    if rank == 0: # main node
        globP = np.zeros(shape = (N, 4, 2), dtype = 'double')
        coreP = np.zeros(shape = (N//size, 4, 2), dtype = 'double')
        for n in range(0, N//size):
            coreP[n, :, :] = init_1P()

        comm.Gather(coreP, globP, root = 0)
        comm.Barrier()
        return globP

    if rank != 0: # satellite node
        # globP = None
        coreP = np.zeros(shape = (N//size, 4, 2), dtype = 'double')
        for n in range(0, N//size):
            coreP[n, :, :] = init_1P()

        # comm.Gather(coreP, globP, root = 0)
        comm.Barrier()
        # return globP
        return None

if __name__ == '__main__':
    # runtime parameters
    M = 256
    N = 8
    nu = 9e-4
    beta = 1
    gamma = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rng = np.random.default_rng(seed = 123)

    x, y, x_grid, y_grid, dx, dy = init_grid(M)
    kp, kq, kU = init_wavearrays(M)

    if rank == 0: # main node
        Omega = init_Omega(beta)

    globP = init_P()
    print(globP)