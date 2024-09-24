# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import matplotlib.pyplot as plt
import numpy as np
from FFTW_interpolate import inter_1D


time = [0, 0.1257, 0.2513, 0.3770, 0.5027, 0.6283, 0.7540, 0.8796, 1.0053, 1.1310, 1.2566, 1.3823, 1.5080, 1.6336, 1.7593, 1.8850, 2.0106, 2.1363, 2.2619, 2.3876, 2.5133, 2.6389, 2.7646, 2.8903, 3.0159, 3.1416, 3.2673, 3.3929, 3.5186, 3.6442, 3.7699, 3.8956, 4.0212, 4.1469, 4.2726, 4.3982, 4.5239, 4.6496, 4.7752, 4.9009, 5.0265, 5.1522, 5.2779, 5.4035, 5.5292, 5.6549, 5.7805, 5.9062, 6.0319, 6.1575]

signal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

plt.plot(time, signal)
t_inter = []
s_inter = []
for ti in np.linspace(0, time[-1] + time[1], 500):
   t_inter.append(ti)
   s_inter.append(inter_1D(time, signal, ti))

plt.plot(t_inter, s_inter, color = 'red')
plt.show()