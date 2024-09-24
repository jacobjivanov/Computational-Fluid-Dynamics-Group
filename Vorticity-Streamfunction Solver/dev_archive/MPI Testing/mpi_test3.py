from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 8

# Define the data to be gathered
send_data = np.random.rand(N//size, 4, 2)

# Prepare to receive data on the root process
if rank == 0:
    recv_data = np.empty(shape = (N, 4, 2), dtype = 'double')
else:
    recv_data = None

# Gather data from all processes onto the root process
comm.Gather(send_data, recv_data, root=0)

# Print gathered data on the root process
if rank == 0:
    print("Gathered data on rank 0:\n", recv_data)