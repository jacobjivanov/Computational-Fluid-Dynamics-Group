import ffti_v6 as fi
import li_v6 as li
import numpy as np

NI = np.linspace(10, 25, (25 - 10) + 1)
# print(NI)
E_fi = [np.zeros(len(NI)), np.zeros(len(NI))]
E_li = [np.zeros(len(NI)), np.zeros(len(NI))]

for i in range(len(NI)):
   x = np.linspace(0, 2 * np.pi, int(NI[i]))
   y = np.e ** np.sin(x)
   
   x_p = 2 * np.pi * np.random.rand(10)
   e_fi = []
   e_li = []
   for p in range(len(x_p)):
      e_fi.append(fi.inter_1D(x, y, x_p[p]) - np.e ** np.sin(x_p[p]))
      e_li.append(li.inter_1D(x, y, x_p[p]) - np.e ** np.sin(x_p[p]))
   
   E_fi[0][i], E_fi[1][i] = np.linalg.norm(e_fi), np.std(e_fi)
   E_li[0][i], E_li[1][i] = np.linalg.norm(e_li), np.std(e_li)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
ax.errorbar(NI, E_fi[0], yerr = E_fi[1], capsize = 5, color = 'blue', linestyle = '', errorevery = 1, alpha = 0.2, label = 'n = 10 Standard\nDeviation of Signed Error')
ax.loglog(NI, E_fi[0], color = 'blue', label = "Trifourier Interpolation Error")

ax.errorbar(NI, E_li[0], yerr = E_li[1], capsize = 5, color = 'red', linestyle = '', errorevery = 1, alpha = 0.2, label = 'n = 10 Standard\nDeviation of Signed Error')
ax.loglog(NI, E_li[0], color = 'red', label = "Trilinear Interpolation Error")

ax.set_title("Error Convergence of inter_1D()")
ax.set_xlabel("Ni")
ax.set_ylabel("2-Norm of Signed Interpolation Error")
ax.set_ylim(1e-16, 1)
ax.legend()
plt.show()
fig.savefig('inter_1D() Error Convergence', dpi = 600)