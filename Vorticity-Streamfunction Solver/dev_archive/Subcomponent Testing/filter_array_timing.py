import numpy as np

M, N = 512, 512

a, b, c, d, = 10, 25, 438, 500

for i in range(0, 10000):
    array = np.random.rand(M, N)
    array[a:b, c:d] = 0

"""
fil = np.ones(shape = (M, N))
fil[a:b, c:d] = 0
for i in range(0, 10000):
    array = np.random.rand(M, N)
    fil_array = fil * array
"""