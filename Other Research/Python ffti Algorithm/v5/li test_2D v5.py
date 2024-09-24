# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import time
t_init = time.time()

import matplotlib.pyplot as plt
import numpy as np
import li_v5 as li

# PART OF EARLIER TESTING, IGNORE
# import linear_interpolate as li

x_max = 2 * np.pi
y_max = 2 * np.pi

Ni = 100 # typically a round number + 1,
Nj = 100 # such that the interstitial values are nice

x, y = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj)

def func(x, y):
   z = np.e ** (np.sin(x) + np.sin(y))
   
   # OTHER EXAMPLE FUNCTIONS
   # z = np.e ** np.sin(x ** 2) + np.log(y + 1)
   # z = x - y ** 1.2
   # z = np.sin(x ** y) - (x * y * np.cos(y / (x + 1)))

   return z

z = np.zeros(shape = (Ni, Nj))
for i in range(0, Ni):
   for j in range(0, Nj):
      z[i, j] = func(x[i], y[j])
      print("Building Data. Progress: {0:07.3f}% Complete".format(100 * (i + (j + 1) / Nj) / Ni), end = '\r')

t_build = time.time()
print("Building Data. Progress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_build - t_init))

Ni_inter = 4 * Ni
Nj_inter = 4 * Nj

x_inter, y_inter = np.linspace(0, x_max, Ni_inter), np.linspace(0, y_max, Nj_inter)

z_li_approx = np.zeros(shape = (Ni_inter, Nj_inter))
z_li_error = np.zeros(shape = (Ni_inter, Nj_inter))

# PART OF TESTING, IGNORE
# z_li_order_diff = np.zeros(shape = (Ni_inter, Nj_inter))
# z_li_approx = np.zeros(shape = (Ni_inter, Nj_inter))
# z_li_error = np.zeros(shape = (Ni_inter, Nj_inter))

for i in range(0, Ni_inter):
   for j in range(0, Nj_inter):
      z_li_approx[i, j] = li.inter_2D(x, y, z, [x_inter[i], y_inter[j]])
      
      z_li_error[i][j] = z_li_approx[i][j] - func(x_inter[i], y_inter[j])

      # PART OF TESTING, IGNORE
      # z_li_order_diff[i][j] = li.inter_2D(x, y, z, [x_inter[i][0], y_inter[0][j]], order = 'xy') - li.inter_2D(x, y, z, [x_inter[i][0], y_inter[0][j]], order = 'yx')

      # z_li_approx[i][j] = li.inter_2D(x, y, z, [x_inter[i][0], y_inter[0][j])
      # z_li_error[i][j] = z_li_approx[i][j] - func(x_inter[i][0], y_inter[0][j)

      # The following print command increases computation time of this double-for loop by approximately 2.7%
      print("Interpolating. Progress: {0:07.3f}% Complete".format(100 * (i + (j + 1) / Nj_inter) / Ni_inter), end = '\r')

t_inter = time.time()
print("Building Data. Progress: 100.000% Complete\t Process Time: {0:10.5f} s".format(t_inter - t_build))

# PLOT SETTINGS BELOW
import matplotlib.pyplot as plt

PLOT_STYLE = "3D SURFACE"
PLOT_STYLE = "2D CONTOUR"

if PLOT_STYLE == "3D SURFACE":
   fig = plt.figure()
   fig.tight_layout()
   ax = fig.add_subplot(projection = '3d')
   
   # Discrete Input Scatter Plot
   ax.scatter(np.meshgrid(x, y, indexing = "ij")[0], np.meshgrid(x, y, indexing = "ij")[1], z, color = "grey", alpha = 1, label = "discrete input")

   # Fourier Interpretation Surface Plot
   ax.plot_surface(x_inter, y_inter, z_li_approx, cmap = "cool", alpha = 0.8)

   # Linear Interpretation Surface Plot
   # ax.plot_surface(x_inter, y_inter, z_li_approx, cmap = "cool", alpha = 0.4)
   # Fourier Interpretation Error Surface Plot
   # ax.plot_surface(x_inter, y_inter, z_li_error, cmap = "jet", alpha = 0.4)
   # Linear Interpretation Error Surface Plot
   # ax.plot_surface(x_inter, y_inter, z_li_error, cmap = "cool", alpha = 0.4)
   # Fourier Interpretation Order Difference Surface Plot
   # ax.plot_surface(x_inter, y_inter, z_li_order_diff, cmap = "cool", alpha = 0.4)

   ax.set_xlabel("x-axis")
   ax.set_ylabel("y-axis")
   ax.set_zlabel("z-axis")

   # ax.legend()

   ax.view_init(azim = 225, elev = 0)
   plt.show()

if PLOT_STYLE == "2D CONTOUR":
   # "on domain 2 pi x 2 pi plot exp(sin(x) + cos(y) ) using M x N points (choose any M and N you like)"
   plt.clf()
   plt.contourf(np.meshgrid(x, y, indexing = 'ij')[0], np.meshgrid(x, y, indexing = 'ij')[1], z, cmap = 'jet', levels = 100)
   plt.colorbar()
   plt.xlabel('x-axis')
   plt.ylabel('y-axis')
   plt.title(r"$e^{\sin(x) + \cos(y)}$" + " on Ni: {0}, Nj: {1}".format(Ni, Nj))
   # plt.savefig("BULLET 1.png", dpi = 600)
   plt.show()


   # "on domain 2 pi x 2 pi use your interpolation to plot exp(sin(x) + cos(y) ) using 4M x 4N points"
   plt.clf() # clears previous
   plt.contourf(np.meshgrid(x_inter, y_inter, indexing = 'ij')[0], np.meshgrid(x_inter, y_inter, indexing = 'ij')[1], z_li_approx, cmap = 'jet', levels = 100)
   plt.colorbar()
   plt.xlabel('x-axis')
   plt.ylabel('y-axis')
   plt.title("Interpolation of " + r"$e^{\sin(x) + \cos(y)}$" + " on Ni_inter: {0}, Nj_inter: {1}".format(Ni_inter, Nj_inter))
   # plt.savefig("BULLET 2.png", dpi = 600)
   plt.show()

   # "on domain 2 pi x 2 pi plot the error between your interpolated function and the exact (this is a plot of the signed error = interpolated - exact)"
   plt.clf() # clears previous
   plt.contourf(np.meshgrid(x_inter, y_inter, indexing = 'ij')[0], np.meshgrid(x_inter, y_inter, indexing = 'ij')[1], z_li_error, cmap = 'jet', levels = 100)
   plt.colorbar()
   plt.xlabel('x-axis')
   plt.ylabel('y-axis')
   plt.title("Signed Error Between Interpolation of " + r"$e^{\sin(x) + \cos(y)}$" + "\non Ni_inter: {0}, Nj_inter: {1} and Exact Solution".format(Ni_inter, Nj_inter))
   # plt.savefig("BULLET 3.png", dpi = 600)
   plt.show()