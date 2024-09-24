import numpy as np
from RK3 import RK3

def lorentz(t, u, c):
    # u = np.array([x, y, z])
    # c = np.array([beta, rho, sigma])

    return np.array([c[2] * (u[1] - u[0]), u[0] * (c[1] - u[2]) - u[1], u[0] * u[1] - c[0] * u[2]])

t = np.linspace(0, 40, 1000)
c = np.array([8/3, 15, 25])
u0 = np.array([1, 1, 1])

u = RK3(lorentz, u0, t, c)

x = np.empty(len(t))
y = np.empty(len(t))
z = np.empty(len(t))

for i in range(len(t)):
    x[i] = u[i, 0]
    y[i] = u[i, 1]
    z[i] = u[i, 2]

import matplotlib.pyplot as plt
fig1 = plt.figure()
fig1.suptitle(r"Lorentz System, $\beta = \frac{8}{3}$, $\rho = 26$, $\sigma = 11$\nSimulation & Animation done by Jacob Ivanov, Researcher for UConn CFDG ")
ax1 = fig1.add_subplot(projection = '3d')
# img1 = ax1.plot(x, y, z, color = 'blue', alpha = 0.2)
img1 = ax1.scatter(x, y, z, c = t, cmap = 'ocean', alpha = 0.2)
fig1.colorbar(img1, location = 'left', label = r'$t$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')
plt.show()