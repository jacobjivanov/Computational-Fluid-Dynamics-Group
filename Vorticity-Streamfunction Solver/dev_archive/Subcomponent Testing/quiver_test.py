import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

M, N = 128, 128
x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)

u = np.empty(shape = (M, N)) # physcial velocity x-component
v = np.empty(shape = (M, N)) # physical velocity y-component
for i in range(0, M):
    for j in range(0, N):
        u[i, j] = + np.cos(x[i]) * np.sin(y[j])
        v[i, j] = - np.sin(x[i]) * np.cos(y[j])

eo = M // 16
fig, ax = plt.subplots(1, 3, constrained_layout = True, figsize = (8, 5))

ax[0].set_title(r"$u$")
u_plot = ax[0].pcolormesh(x, y, u.T)

ax[1].set_title(r"$\langle u, v \rangle$")
uv_plot = ax[1].quiver(x[::eo], y[::eo], u[::eo, ::eo].T, v[::eo, ::eo].T, angles = 'xy')

ax[2].set_title(r"$v$")
v_plot = ax[2].pcolormesh(x, y, v.T)

"""
for plt_i in [0, 1, 2]:
    ax[plt_i].set_aspect('equal')

    ax[plt_i].set_xlim(0, 2 * np.pi)
    ax[plt_i].set_ylim(0, 2 * np.pi)

    ax[plt_i].set_xlabel(r"$x$")
    ax[plt_i].set_ylabel(r"$y$")

    ax[plt_i].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax[plt_i].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax[plt_i].xaxis.set_major_locator(MultipleLocator(base = np.pi))
    ax[plt_i].yaxis.set_major_locator(MultipleLocator(base = np.pi))

for plot in [u_plot, uv_plot, v_plot]:
    divider = make_axes_locatable(plot.axes)
    cax = divider.append_axes("bottom", size = "5%", pad = 0.3)
    fig.colorbar(plot, cax = cax, orientation = 'horizontal')
"""

plt.show()