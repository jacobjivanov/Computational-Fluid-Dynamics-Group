import numpy as np
import ffti_v5 as fi

x_max, y_max, z_max = 10, 10, 10
Ni, Nj, Nk = 101, 101, 101

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

rho = np.zeros(shape = (Ni, Nj, Nk),)# dtype = 'float64')

def func(x, y, z):
   rho = np.sin(x ** 2) + np.log(y + 1) + z
   # rho = np.sin(x / 10 * y / 10 * z / 10)
   return rho

for i in range(0, Ni): 
   for j in range(0, Nj): 
      for k in range(0, Nk):
         rho[i, j, k] = func(x[i], y[j], z[k])

      print("Building Data. Progress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj) ) / Ni), end = '\r')
print()

x_plot, y_plot, z_plot = [], [], []
rho_app, rho_ana, rho_diff = [], [], []
c_plot, a_plot = [], []

p, pt = 0, 100
while p < pt:
   x_plot.append(np.random.random() * x_max)
   y_plot.append(np.random.random() * y_max)
   z_plot.append(np.random.random() * z_max)

   rho_app.append(fi.inter_3D(x, y, z, rho, pos = [x_plot[-1], y_plot[-1], z_plot[-1]]))
   rho_ana.append(func(x_plot[-1], y_plot[-1], z_plot[-1]))
   
   rho_diff.append(rho_ana[-1] - rho_app[-1])
   
   print("Interpolating. Progress: {0:07.3f}% Complete".format(100 * (p + 1) / pt), end = '\r')
   p += 1

import matplotlib.pyplot as plt

fig = plt.figure()
fig.tight_layout()
ax = fig.add_subplot(projection = '3d')

ax.scatter(x_plot, y_plot, z_plot, c = rho_diff, cmap = 'seismic_r', alpha = 1)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

ax.view_init(azim = 225, elev = 0)

# plt.colorbar(rho_app)
plt.show()