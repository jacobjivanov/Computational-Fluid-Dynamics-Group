import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 8

if rank == 0: # main node
    P = np.empty(shape = (N, 3, 2))
    for n in range(0, size):
        P[n, :, :] = n
    
    coreP = np.empty(shape = (N//size, 3, 2), dtype = 'double')
    comm.Scatter(P, coreP)
    
    comm.barrier()
    print("core {0}/{1}, coreP:\n {2}\n".format(rank, size, coreP))

if rank != 0: # satellite node
    P = None
    coreP = np.empty(shape = (N//size, 3, 2), dtype = 'double')
    comm.Scatter(P, coreP)
    
    comm.barrier()
    print("core {0}/{1}, coreP:\n {2}\n".format(rank, size, coreP))
