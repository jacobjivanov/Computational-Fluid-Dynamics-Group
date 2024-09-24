import numpy as np
import matplotlib.pyplot as plt
import li_v5 as li

x = np.linspace(0, 10, 12)
y = np.random.rand(12)

x_inter = np.linspace(0, 10, 1000)
y_inter = np.zeros(1000)

for i in range(len(x_inter)):
   y_inter[i] = li.inter_1D(x, y, x_inter[i])

plt.scatter(x, y)
plt.plot(x_inter, y_inter)
plt.show()
