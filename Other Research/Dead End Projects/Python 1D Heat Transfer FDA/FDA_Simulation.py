# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. 

# A composite wall separates combustion gases at 2400 °C from a liquid coolant at 20 °C. The wall is composed of a 10-mm-thick layer of beryllium oxide on the gas side and a 20-mm-thick slab of stainless steel (AISI 304) on the liquid side.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import thermophysical_properties as tp

dx = 0.1 / 1000 # m
dt = 0.0001 # s
t_max = 100 # s

Tb_x0 = 2673.15 # K
Tb_x30 = 293.15 # K

def Fo(x, T):
   # print(x * 1000, T, alpha(x, T) * dt / dx ** 2)
   return tp.alpha(x, T) * dt / dx ** 2

init_T = np.zeros(int(0.030 / dx) + 1)
x = np.linspace(0, 0.030, int(0.030 / dx + 1))
# print(x)
for xi in range(len(x)):
   # print(xi, alpha(xi * dx, 0))
   # xi = 2380 * np.random.random() + 293.15
   init_T[xi] = 2380 * np.random.random() + 293.15
   # init_T[xi] = 1000
   # Tb_x0 - (Tb_x0 - Tb_x30) / 0.030 * xi * dx

curr_T = init_T.copy()

curr_alpha = np.zeros(len(init_T))
curr_k = np.zeros(len(init_T))
curr_cP = np.zeros(len(init_T))
curr_rho = np.zeros(len(init_T))

# Prevents repetative curr_rho setting, since it doesn't change
for xi in range(len(curr_T)):
   curr_rho[xi] = tp.rho(xi * dx, curr_T[xi])

def next_T(now_T):
   next_T = now_T.copy()
   
   global curr_alpha, curr_k, curr_rho, curr_cP

   for xi in range(len(now_T)):
      xi_Fo = Fo(xi * dx, now_T[xi])
      
      # Current Thermophysical Properties
      curr_alpha[xi] = tp.alpha(xi * dx, now_T[xi])
      curr_k[xi] = tp.k(xi * dx, now_T[xi])
      curr_cP[xi] = tp.cP(xi * dx, now_T[xi])
      
      # Unecessary to repeat
      # curr_rho[xi] = tp.rho(xi * dx, now_T[xi])

      # print(xi, "{0:.2f}".format(xi_Fo))
      if xi == 0:
         # print("case1", xi)
         next_T[xi] = xi_Fo * (now_T[xi + 1] + Tb_x0) + (1 - 2 * xi_Fo) * now_T[xi]
      elif xi == len(now_T) - 1:
         # print("case2", xi)
         next_T[xi] = xi_Fo * (Tb_x30 + now_T[xi - 1]) + (1 - 2 * xi_Fo) * now_T[xi]
      else: 
         # print("case3", xi)
         next_T[xi] = xi_Fo * (now_T[xi + 1] + now_T[xi - 1]) + (1 - 2 * xi_Fo) * now_T[xi]
      
   global curr_T
   curr_T = next_T.copy()
   return next_T

fig, ax = plt.subplots(2, 1, figsize = (10, 10), tight_layout = True)
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9],) 
plt.subplots_adjust(hspace = 0.3, right = 0.7)

# Enables secondary y-axis for each additional thermophysical property, adjusts position
ax_k = ax[1].twinx()
ax_cP = ax[1].twinx()
ax_rho = ax[1].twinx()

ln1, = ax[0].plot([], [])
ln2, = ax[1].plot([], [])

def init(): return ln1, ln2,

def update(t):
   # print(t)
   new_T = next_T(curr_T)

   ax[0].clear()
   ax[0].plot(x, init_T, color = "black", linestyle = "dashed", alpha = 0.1, label = "Initial Temperature Distribution")
   ax[0].plot(x, new_T, color = "black", label = "Current Temperature Distribution")
   ax[0].axvline(0.010, color = 'grey', linestyle = 'dotted')
   ax[0].set_xlim(0, 0.030) # m
   ax[0].set_ylim(0, 3000) # K
   ax[0].set_xlabel("x Position (m)")
   ax[0].set_ylabel("Temperature (K)")
   ax[0].set_title("Temperature")
   ax[0].legend()

   ax[1].clear()
   # ax[1].scatter(x, curr_alpha)
   ax[1].plot(x, curr_alpha, color = "blue")
   ax[1].set_xlim(0, 0.030) # m
   ax[1].set_ylim(0, 1e-5)
   ax[1].set_xlabel("x Position (m)")
   ax[1].set_ylabel(r"Thermal Diffusivity $\left(\frac{\mathrm{m}^2}{\mathrm{s}}\right)$")
   ax[1].set_title("Thermophysical Properties")
   ax[1].tick_params(axis = 'y', colors = "blue")
   ax[1].yaxis.label.set_color("blue")

   ax_k.clear()
   ax_k.tick_params(colors = "red")
   ax_k.set_ylabel(r"Thermal Conductivity $\left(\frac{\mathrm{W}}{\mathrm{m \cdot K}}\right)$")
   ax_k.yaxis.label.set_color("red")
   ax_k.set_ylim(0, 50)
   ax_k.plot(x, curr_k, color = "red", linestyle = "dotted")

   ax_cP.clear()
   ax_cP.spines.right.set_position(("axes", 1.15))
   ax_cP.yaxis.label.set_color("orange")
   ax_cP.tick_params(colors = "orange")
   ax_cP.set_ylabel(r"Isobaric Specific Heat $\left(\frac{\mathrm{kJ}}{\mathrm{kg \cdot K}}\right)$")
   ax_cP.plot(x, curr_cP / 1000, color = "orange", linestyle = "dotted")

   ax_rho.clear()
   ax_rho.spines.right.set_position(("axes", 1.3))
   ax_rho.tick_params(colors = "green")
   ax_rho.set_ylabel(r"Density $\left(\frac{\mathrm{kg}}{\mathrm{m}^3}\right)$")
   ax_rho.yaxis.label.set_color("green")
   ax_rho.set_ylim(0, 10000)
   ax_rho.plot(x, curr_rho, color = "green", linestyle = "dotted")

   fig.suptitle("1D Conduction Simulation\n t = {0:.4f} s\n".format(t))
   for t_save in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]:
      if (t_save - dt / 2 < t) & (t < t_save + dt / 2):
         fig.savefig("/Users/jacobivanov/Desktop/University of Connecticut/Computational Fluid Dynamics Group/1D Heat Transfer/Frames/FDA Simulation t = {0:.4f} s.jpg".format(t), dpi = 300)
   fig.show()
   return ln1, ln2

anim = ani.FuncAnimation(fig, update, init_func = init, frames = np.linspace(0, t_max, int(t_max / dt + 1)), blit = False, repeat = False)
plt.show()

'''
plt.rcParams['animation.ffmpeg_path'] = "/opt/homebrew/bin/ffmpeg"
plt.rcParams['animation.convert_path'] = "/opt/homebrew/bin/convert"
'''

# ani.FileMovieWriter(fps = 30, "/Users/jacobivanov/Desktop/University of Connecticut/Computational Fluid Dynamics Group/1D Heat Transfer/FDA Simulation.jpg")

# plt.show()