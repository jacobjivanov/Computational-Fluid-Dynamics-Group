# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. 

# A composite wall separates combustion gases at 2400 °C from a liquid coolant at 20 °C. The wall is composed of a 10-mm-thick layer of beryllium oxide on the gas side and a 20-mm-thick slab of stainless steel (AISI 304) on the liquid side.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import thermophysical_properties as tp

dx = 1 / 1000 # m
dt = 0.01 # s
t_max = 100 # s

Tb_x0 = 2673.15 # K
Tb_x30 = 293.15 # K

init_T = np.zeros(int(0.030 / dx) + 1)
x = np.linspace(0, 0.030, int(0.030 / dx + 1))
for xi in range(len(x)):
   # init_T[xi] = 2380 * np.random.random() + 293.15
   init_T[xi] = 2000

curr_T = init_T.copy()

def Fo(x, T):
   if (x <= 0.010):
      return 3.72e-07 * dt / dx ** 2
   else:
      return 5.25e-06 * dt / dx ** 2

# This function is structured kind of dumb because that's the required way for matplotlib animations to work
def next_T(now_T):
   next_T = now_T.copy()

   for xi in range(len(now_T)):
      xi_Fo = Fo(xi * dx, now_T[xi])
      
      if xi == 0:
         next_T[xi] = xi_Fo * (now_T[xi + 1] + Tb_x0) + (1 - 2 * xi_Fo) * now_T[xi]
      elif xi == len(now_T) - 1:
         next_T[xi] = xi_Fo * (Tb_x30 + now_T[xi - 1]) + (1 - 2 * xi_Fo) * now_T[xi]
      else: 
         next_T[xi] = xi_Fo * (now_T[xi + 1] + now_T[xi - 1]) + (1 - 2 * xi_Fo) * now_T[xi]
      
   global curr_T
   curr_T = next_T.copy()
   return next_T

t = 0
while t <= t_max:
   next_T(curr_T)
   t += dt

plt.plot(x, curr_T)
plt.show()