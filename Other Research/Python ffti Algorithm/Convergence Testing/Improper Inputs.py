import numpy as np
import ffti_v6 as fi
import matplotlib.pyplot as plt

def f1(x):
   if x < np.pi: return 0.00
   else: return 1.00

def f2(x):
   return np.abs(np.sin(x) + 0.25)

x1 = [np.linspace(0, 2 * np.pi, 10), np.linspace(0, 2 * np.pi, 50), np.linspace(0, 2 * np.pi, 200)]
y1 = [np.zeros(10), np.zeros(50), np.zeros(200)]

x1_inter = np.linspace(0, 2 * np.pi, 1000)
y1_inter = [np.zeros(1000), np.zeros(1000), np.zeros(1000)]

for d in range(len(x1)):
   for i in range(len(x1[d])):
      y1[d][i] = f1(x1[d][i])

for d in range(len(y1_inter)):
   for i in range(len(x1_inter)):
      y1_inter[d][i] = fi.inter_1D(x1[d], y1[d], x1_inter[i])

x2 = [np.linspace(0, 2 * np.pi, 10), np.linspace(0, 2 * np.pi, 50), np.linspace(0, 2 * np.pi, 200)]
y2 = [np.zeros(10), np.zeros(50), np.zeros(200)]

x2_inter = np.linspace(0, 2 * np.pi, 1000)
y2_inter = [np.zeros(1000), np.zeros(1000), np.zeros(1000)]

for d in range(len(x2)):
   for i in range(len(x2[d])):
      y2[d][i] = f2(x2[d][i])

for d in range(len(y2_inter)):
   for i in range(len(x2_inter)):
      y2_inter[d][i] = fi.inter_1D(x2[d], y2[d], x2_inter[i])

fig, ax = plt.subplots(2, 3, figsize = (16, 9))
ax[0][0].plot(x1_inter, y1_inter[0], color = "blue", label = "1D Fourier\nInterpolation")
ax[0][0].scatter(x1[0], y1[0], color = 'black', label = "Discrete Input")
ax[0][0].legend()
ax[0][0].set_ylabel("y")
ax[0][0].set_title("Discontinuous: N = 10")

ax[0][1].plot(x1_inter, y1_inter[1], color = "blue", label = "1D Fourier\nInterpolation")
ax[0][1].scatter(x1[1], y1[1], color = 'black', label = "Discrete Input")
ax[0][1].legend()
ax[0][1].set_title("Discontinuous: N = 50")


ax[0][2].plot(x1_inter, y1_inter[2], color = "blue", label = "1D Fourier\nInterpolation")
ax[0][2].scatter(x1[2], y1[2], linestyle = 'dashed', color = 'black', label = "Discrete Input")
ax[0][2].legend()
ax[0][2].set_title("Discontinuous: N = 200")


ax[1][0].plot(x2_inter, y2_inter[0], color = "blue", label = "1D Fourier\nInterpolation")
ax[1][0].scatter(x2[0], y2[0], color = 'black', label = "Discrete Input")
ax[1][0].set_ylabel("y")
ax[1][0].set_xlabel("x")
ax[1][0].set_title("Non-Continuously Differentiable: N = 10")
ax[1][0].legend()

ax[1][1].plot(x2_inter, y2_inter[1], color = "blue", label = "1D Fourier\nInterpolation")
ax[1][1].scatter(x2[1], y2[1], color = 'black', label = "Discrete Input")
ax[1][1].set_xlabel("x")
ax[1][1].set_title("Non-Continuously Differentiable: N = 50")
ax[1][1].legend()

ax[1][2].plot(x2_inter, y2_inter[2], color = "blue", label = "1D Fourier\nInterpolation")
ax[1][2].scatter(x2[2], y2[2], linestyle = 'dashed', color = 'black', label = "Discrete Input")
ax[1][2].set_xlabel("x")
ax[1][2].set_title("Non-Continuously Differentiable: N = 200")
ax[1][2].legend()

plt.show()
fig.savefig("Improper Inputs.png", dpi = 600)