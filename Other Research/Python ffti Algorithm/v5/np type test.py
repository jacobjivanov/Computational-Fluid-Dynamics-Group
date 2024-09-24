import ffti_v6 as fi
import numpy as np
from time import time

x_max, y_max, z_max = 2 * np.pi, 2 * np.pi, 2 * np.pi
Ni, Nj, Nk = 21, 21, 21

x, y, z = np.linspace(0, x_max, Ni), np.linspace(0, y_max, Nj), np.linspace(0, z_max, Nk)

print(type(x[2]))
print(x[2])