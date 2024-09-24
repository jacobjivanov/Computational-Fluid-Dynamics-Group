import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

tau = 0.49
beta = 1
M, N = 256, 256
nu = 9e-4
t_end = 200
dt = 0.1

pos_p = np.array([np.pi + 0.01, np.pi + 0.01])
vel_p = np.zeros(2)
vel_f = np.zeros(2)

def VEL_F(t, x, y):
    return np.array([
        np.exp(-2 * nu * t) * np.cos(beta * x) * np.sin(beta * y), 
        - np.exp(-2 * nu * t) * np.sin(beta * x) * np.cos(beta * y)
    ])

def ACC_F(t, x, y):
    return np.array([
        -2 * nu * np.exp(-2 * nu * t) * np.cos(beta * x) * np.sin(beta * y), 
        2 * nu * np.exp(-2 * nu * t) * np.sin(beta * x) * np.cos(beta * y)
    ])

fig, ax = plt.subplots(1, 1)
pos_plot = ax.scatter(pos_p[0], pos_p[1], c = 0, cmap = 'bwr', vmin = 0, vmax = 200)

t, n = 0, 0
while t < t_end:
    pos_star = pos_p + dt * vel_p
    
    vel_f_star = VEL_F(t, *pos_p)
    
    vel_p_star = vel_p + dt/tau * (vel_f_star + 3 * tau * ACC_F(t, *pos_p))
    vel_p_star /= 1 + dt/tau

    pos_p2 = pos_p + dt/2 * (vel_p + vel_p_star)

    vel_p = 1/(1 + dt/(2 * tau)) * (vel_p + dt/2 * (3 * (ACC_F(t, *pos_p) + ACC_F(t + dt, *pos_p2))) + (VEL_F(t, *pos_p) + VEL_F(t + dt, *pos_p2) - vel_p)/tau)
    
    pos_p = pos_p2 % (2 * np.pi)
    print(t)
    pos_plot = ax.scatter(pos_p[0], pos_p[1], c = t, cmap = 'bwr', vmin = 0, vmax = 200)
    n += 1
    t += dt

pos_cbar = fig.colorbar(pos_plot)
pos_cbar.set_label(r"$t$")

ax.set_aspect('equal')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.xaxis.set_major_locator(MultipleLocator(base = np.pi))
ax.yaxis.set_major_locator(MultipleLocator(base = np.pi))

fig.suptitle("Particle Position over Time, " + r"$\tau = $" + "{0}".format(tau), y = 0.95)

plt.savefig("Inertial Particle tau = {0}.png".format(tau), dpi = 200)