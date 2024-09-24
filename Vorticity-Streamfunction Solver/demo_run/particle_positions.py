# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant for Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. Contact me jacob.ivanov@uconn.edu for any questions/issues.

# The goal of this program is to evolve the vorticity transport equation initialized as perturbed Taylor-Green Vortex Flow. Additionally, variably inertial particles will be trasported within the flow. This version has been written to be run in the CLI.

import numpy as np
from numpy import real, imag
from numpy.fft import fft2, ifft2

import numba
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

s_min = 15000
s_max = 25000

s_int = 5
N = 10
savepath = '512 10 1 True'

data = np.load("data/" + savepath + "/v9 data s = {0}.npz".format(s_min))
t = data['t']
print(t)
data = np.load("data/" + savepath + "/v9 data s = {0}.npz".format(s_max))
t = data['t']
print(t)


if __name__ == '__main__':
    
    P_trails = np.zeros(shape = (s_max//s_int, N, 2))

    fig, ax = plt.subplots(1, 1, constrained_layout = False)
    ax.set_aspect('equal')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val,pos: r'{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax.yaxis.set_major_locator(MultipleLocator(base = np.pi))

    for s in range(s_min, s_max, s_int):
        data = np.load("data/" + savepath + "/v9 data s = {0}.npz".format(s))
        P = data['P']

        P_trails[s//s_int, :, 0] = P[:, 0, 0]
        P_trails[s//s_int, :, 1] = P[:, 0, 1]
    
    for n in range(0, N):
        ax.scatter(P_trails[::5, n, 0], P_trails[::5, n, 1], s = 0.5, alpha = 1)

    ax.set_title(r"Particle Pathlines, $t \in [72.69, 141.39]$")
    plt.savefig("anim/Particle Pathlines.png", dpi = 200)
    plt.show()