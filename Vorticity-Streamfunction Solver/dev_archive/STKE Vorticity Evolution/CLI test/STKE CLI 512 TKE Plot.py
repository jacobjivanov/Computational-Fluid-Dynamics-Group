import numpy as np
import pandas as pd

df_CLI0 = pd.read_csv("STKE Vorticity Evolution/CLI test/CLI0 512.csv")
df_CLI1 = pd.read_csv("STKE Vorticity Evolution/CLI test/CLI1 512.csv")
df_CLI2 = pd.read_csv("STKE Vorticity Evolution/CLI test/CLI2 512.csv")
df_CLI3 = pd.read_csv("STKE Vorticity Evolution/CLI test/CLI3 512.csv")
df_CLI4 = pd.read_csv("STKE Vorticity Evolution/CLI test/CLI4 512.csv")

n0 = np.array(df_CLI0['n'])
n1 = np.array(df_CLI1['n'])
n2 = np.array(df_CLI2['n'])
n3 = np.array(df_CLI3['n'])
n4 = np.array(df_CLI4['n'])

tke0 = np.array(df_CLI0['TKE'])
tke1 = np.array(df_CLI1['TKE'])
tke2 = np.array(df_CLI2['TKE'])
tke3 = np.array(df_CLI3['TKE'])
tke4 = np.array(df_CLI4['TKE'])

import matplotlib.pyplot as plt
plt.semilogy(n0, tke0, label = 'CLI0')
plt.semilogy(n1, tke1, label = 'CLI1')
plt.semilogy(n2, tke2, label = 'CLI2')
plt.semilogy(n3, tke3, label = 'CLI3')
plt.semilogy(n4, tke4, label = 'CLI4')
plt.legend()
plt.xlabel(r"$n$")
plt.ylabel("TKE")
plt.title("TKE vs Timestep for CLI STKE, " + "M = 512, " + r"$\nu$" + " = 2e-4")
plt.ylim([1e10, 1e11])
plt.tight_layout(rect = (0.05, 0.05, 0.95, 0.95))
plt.savefig("TKE Plot, 512.png", dpi = 200)