import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import thermophysical_properties as tp


dx = 1 / 1000 # m

init_T = np.zeros(int(0.030 / dx) + 1)
x = np.linspace(0, 0.030, int(0.030 / dx + 1))
for xi in range(len(x)):
   init_T[xi] = 2380 * np.random.random() + 293.15

curr_alpha = np.zeros(len(init_T))
curr_k = np.zeros(len(init_T))
curr_rho = np.zeros(len(init_T))
curr_cP = np.zeros(len(init_T))

for xi in range(len(init_T)):
   curr_alpha[xi] = tp.alpha(xi * dx, init_T[xi])
   curr_k[xi] = tp.k(xi * dx, init_T[xi])
   curr_rho[xi] = tp.rho(xi * dx, init_T[xi])
   curr_cP[xi] = tp.rho(xi * dx, init_T[xi])

fig, ax = plt.subplots(2, 1, figsize = (10, 10), tight_layout = True)
plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9],) 
plt.subplots_adjust(hspace = 0.3)

ax[0].plot(x, init_T)
ax[1].plot(x, curr_alpha)

ax_cP = ax[1].twinx()
ax_cP.plot(x, curr_cP)

plt.show()