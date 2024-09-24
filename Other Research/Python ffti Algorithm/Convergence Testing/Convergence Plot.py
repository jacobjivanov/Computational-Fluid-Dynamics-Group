import numpy as np
import pandas as pd

df = pd.read_csv("convergence_iteration.csv", header = 0, sep = ',')
# print(df)

Ni = df['Ni'].to_numpy()
error_li = df['li_error'].to_numpy()
error_fi = df['fi_error'].to_numpy()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize = (10.6, 6))

ax.scatter(Ni, error_li, label = "Trilinear Interpolation", color = 'red')
ax.scatter(Ni, error_fi, label = "Trifourier Interpolation", color = 'blue')
ax.axhline(2.22044604925e-16, label = "64 Bit Precision", linestyle = 'dashed', color = 'gray')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title("Error Convergence of inter_3D()")
ax.set_xlabel("Ni")
ax.set_ylabel("2-Norm of Signed Interpolation Error")
ax.set_ylim(1e-16, 1e2)
ax.legend(loc = "center left")

plt.show()
fig.savefig('inter_3D() Error Convergence', dpi = 600)