# mpirun -n 5 python3 MPI_function.py

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(0)

N = 4 # arbitrary
values = np.random.rand(N, 10000)

if rank == 0:
   section = np.zeros(int(N / size + N % size))
   start = 0
else: # rank != 0:
   section = np.zeros(int(N / size))
   start = rank * len(section) + N % size
print("Node: {0}, Section Size: {1}".format(rank, len(section)))

for i in range(len(section)):
   section[i] = np.mean(values[start + i])
   # print(start + i)

if rank != 0:
   comm.Send(section, dest = 0, tag = 14)
else: # rank == 0:
   results = np.pad(section, (0, N - len(section)), constant_values = 0)

   for r in range(1, size):
      temp = np.zeros(int(N / size))
      comm.Recv(temp, source = r, tag = 14)

      start = r * int(N / size) + N % size
      for i in range(int(N / size)):
         results[start + i] = temp[i]
   import matplotlib.pyplot as plt
   plt.plot(results)
   plt.show()