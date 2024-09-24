# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# A composite wall separates combustion gases at 2400 °C from a liquid coolant at 20 °C. The wall is composed of a 10-mm-thick layer of beryllium oxide on the gas side and a 20-mm-thick slab of stainless steel (AISI 304) on the liquid side.

import numpy as np
import matplotlib.pyplot as plt
from thermal_diffusivity import alpha

delta_x = 0.1 # mm, 30 must be divisible by delta_x
delta_t = 0.1 # s

T_x0 = 2673.15 # K
T_x30 = 293.15 # K

T_dist = np.zeros(int(30 / delta_x) + 1)
for x_i in range(len(T_dist)):
   T_dist[x_i] = T_x0 - (T_x0 - T_x30) / 30 * x_i * delta_x


t_i = 0 # s
while t_i <= 1000:
   T_dist[0] = T_x0 # Boundary Condition
   T_dist[-1] = T_x30 # Boundary Condition

   for x_i in range(len(T_dist)):
      if x_i == 0:
         T_left = T_x0
         T_right = T_dist[x_i + 1]
      if x_i == len(T_dist) - 1:
         T_left = T_dist[x_i - 1]
         T_right = T_x30
      else:
         T_right = T_dist[x_i + 1]
         T_left = T_dist[x_i - 1]
      delT = T_left - 2 * T_dist[x_i] + T_right
      T_dist[x_i] = (delta_t / delta_x ** 2) * alpha(x_i * delta_x, T_dist[x_i]) * delT
   t_i += 1


plt.plot(T_dist)
plt.show()