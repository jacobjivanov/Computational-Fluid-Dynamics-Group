import numpy as np
import ffti_v7 as fi
import matplotlib.pyplot as plt

def test1():
   Ni = 30 #10 + int(np.random.random() * 30)
   Nj = 30 #10 + int(np.random.random() * 30)
   x_max = 100 * np.random.random()
   y_max = 100 * np.random.random()
   x = np.linspace(0, x_max, Ni)
   y = np.linspace(0, y_max, Nj)

   # f(x, y) must be periodic within both [0, x_max] and [0, y_max]
   def f(x, y):
      norm_x = x * 2 * np.pi / x_max
      norm_y = y * 2 * np.pi / y_max
      
      return np.e ** (np.sin(norm_x) + np.sin(norm_y))

   values2D = np.zeros(shape = (Ni, Nj))
   for i in range(0, Ni):
      for j in range(0, Nj):
         values2D[i, j] = f(x[i], y[j])
   print(values2D)
   # print(values2D[10, 17]) CHECKS OUT

   Ni_inter = Ni * 4
   Nj_inter = Nj * 4
   
   x_inter = np.linspace(0, x_max, Ni_inter)
   y_inter = np.linspace(0, y_max, Nj_inter)

   values2D_inter = np.zeros(shape = (Ni_inter, Nj_inter))
   values2D_exact = np.zeros(shape = (Ni_inter, Nj_inter))
   values2D_error = np.zeros(shape = (Ni_inter, Nj_inter))
   
   print(Ni, Nj)
   for i in range(0, Ni_inter):
      for j in range(0, Nj_inter):
         values2D_inter[i, j] = fi.inter_2D(x, y, values2D, pos = [x_inter[i], y_inter[j]])
         values2D_exact[i, j] = f(x_inter[i], y_inter[j])
      print(i)
   values2D_error = values2D_inter - values2D_exact

   xy_inter = np.meshgrid(x_inter, y_inter, indexing = 'ij')
   plt.pcolor(xy_inter[0], xy_inter[1], values2D_error, cmap = 'bwr')
   # plt.pcolor(x, y, values2D)
   plt.colorbar()
   plt.show()
   
   """
   print(Ni, x_max)
   xy = np.meshgrid(x, y, indexing = 'ij')
   plt.contourf(xy[0], xy[1], values2D, cmap = 'bwr')
   plt.colorbar()
   plt.show()
   """

def test2():
   x = np.linspace(0, 2 * np.pi, 14)
   y = [0.79379, 0.65885, 0.26912, 0.60397, 0.12430, 0.07876, 0.51430, 0.74956, 0.28334, 0.66792, 0.35352, 0.04800, 0.01439, 0.12871]

   pos = 2.1
   x_i = fi.inter_1D(x, y, pos)
   print()
   print(x_i)
   # print(np.sin(pos))

def test3():
   x = np.linspace(0, 2 * np.pi, 30)
   y = np.array([1.00000000000000, 1.23982524745171, 1.52179278181533, 1.83157126863590, 2.14290429283459, 2.41938177933111, 2.62098445274053, 2.71429815589422, 2.68270743774812, 2.53191699109498, 2.28802499571134, 1.98913418044017, 1.67456571524170, 1.37616621793835, 1.11418034548409, 0.897520768565990, 0.726656407463710, 0.597169756252693, 0.502731293762553, 0.437058162334062, 0.394957656004169, 0.372757754322776, 0.368419363889135, 0.381536028935383, 0.413328730729084, 0.466656398675286, 0.545979300464116, 0.657119689322690, 0.806565281724472, 1.00000000000000])

   pos = 2.1
   x_i = fi.inter_1D(x, y, pos)
   print(y.shape)
   print(x_i)

def test4():
   x = np.linspace(0, 2 * np.pi, 30)
   y = np.array([0.4613, 0.6361, 0.2637, 0.0480, 0.4013, 0.1011, 0.1270, 0.8851, 0.5284, 0.0644, 0.6606, 0.5225, 0.2979, 0.6718, 0.2959, 0.1857, 0.5637, 0.8624, 0.4132, 0.9710, 0.0605, 0.5345, 0.5416, 0.0966, 0.2570, 0.2175, 0.7798, 0.7260, 0.4283, 0.3716])

   pos = 2.12
   x_i = fi.inter_1D(x, y, pos)
   print(y.shape)
   print(x_i)

test1()