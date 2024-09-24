from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    numData = data.size
    comm.send(numData, dest = 1)
    comm.Send(data, dest = 1)

if rank != 0:
    numData = comm.recv(source=0)
    print('Number of data to receive: ',numData)

    data = np.empty(numData, dtype='d')  # allocate space to receive the array
    comm.Recv(data, source=0)

    print('data received: ',data)


# # mpiexec -n 4 python3 mpi_test0.py

# from mpi4py import MPI
# import numpy as np

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# if rank == 0: # main node
#     print("main node")

#     array = np.array([
#         [1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9],
#     ])

#     for r in range(1, 4):
#         comm.send(array[r-1].size, dest = r)
#         comm.Send(array[r-1], dest = r)
#         print(array[r-1])

# if rank != 0: # satellite node
#     ndata = comm.recv(source = 0)
#     sub_array = np.zeros(ndata)
#     comm.Recv(sub_array, source = 0)

#     print("satellite node {0}, data: {1}, {2}".format(rank, sub_array, ndata))