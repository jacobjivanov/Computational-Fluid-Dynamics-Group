# mpirun -n 5 python3 MPI_function.py

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

np.random.seed(0)

N = 20245 # arbitrary
values = np.random.rand(N, 10000)

# defines section sizes (and therefore start positions) to be as fair as possible
def section_size(rank):
   if rank == 0 or rank > N % SIZE:
      return N // SIZE
   else: 
      return N // SIZE + 1

def start_position(rank):
   position = 0
   for r in range(1, rank):
      position += section_size(r)
   return position

section = np.zeros(section_size(RANK))
start = start_position(RANK)

print("Node: {0}, Section Size: {1}, Start: {2}".format(RANK, len(section), start))

for i in range(len(section)):
   section[i] = np.mean(values[start + i])

if RANK != 0 and section_size(RANK) != 0:
   comm.Send(section, dest = 0, tag = 14)
else: # rank == 0:
   results = np.pad(section, (0, N - len(section)), constant_values = 0)

   for r in range(1, SIZE):
      if section_size(RANK) != 0:
         temp = np.zeros(N // SIZE)
         comm.Recv(temp, source = r, tag = 14)

         start = r * (N // SIZE) + N % SIZE
         for i in range(N // SIZE):
            results[start + i] = temp[i]
   import matplotlib.pyplot as plt
   plt.plot(results)
   plt.show()
