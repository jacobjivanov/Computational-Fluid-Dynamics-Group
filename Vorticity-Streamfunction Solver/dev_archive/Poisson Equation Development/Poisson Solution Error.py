import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
# import ffti_v10 as fi


def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

x_dom, y_dom = [0, 2 * pi], [0, 2 * pi]
L_x, L_y = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0]
M, N = 8, 8
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
kx, ky = wavenumbers(M, x_dom), wavenumbers(N, y_dom)

omega = np.zeros(shape = (M, N))
psi = np.zeros(shape = (M, N))

for i in range(0, M):
    for j in range(0, N):
        # Analytical Solution, "Fourier Simple"
        # psi[i, j] = np.sin(x[i]) * np.cos(y[j])
        # omega[i, j] = 2 * psi[i, j]

        # Analytical Solution, "Fourier Simple"
        # psi[i, j] = np.cos(y[j])
        # omega[i, j] = np.cos(y[j])

        # Analytical Solution, "Fourier Simple"
        # psi[i, j] = np.sin(x[i])
        # omega[i, j] = np.sin(x[i])

        # Analytical Solution, Not "Fourier Simple"
        psi[i, j] = np.exp(np.sin(x[i]) + np.cos(y[j]))
        omega[i, j] = - psi[i, j] * (np.cos(x[i]) ** 2 - np.cos(y[j]) - np.sin(x[i]) + np.sin(y[j]) ** 2)

        # Analytical Solution, Not "Fourier Simple"
        # psi[i, j] = np.sin(np.exp(np.sin(x[i])))
        # omega[i, j] = np.exp(np.sin(x[i]))
        # omega[i, j] *= omega[i, j] * np.cos(x[i]) ** 2 * psi[i, j] + (np.sin(x[i]) - np.cos(x[i]) ** 2) * np.cos(np.exp(np.sin(x[i])))

        # Analytical Solution, Not "Fourier Simple"
        # psi[i, j] = np.log(np.sin(x[i]) + 2) # np.log() is defined as base e
        # omega[i, j] = np.sin(x[i])/(np.sin(x[i]) + 2) + (np.cos(x[i]) ** 2)/((np.sin(x[i]) + 2) ** 2)

        # Analytical Solution, Not "Fourier Simple"
        # psi[i, j] = np.exp(np.sin(x[i]))
        # omega[i, j] = (np.sin(x[i]) - np.cos(x[i]) ** 2) * np.exp(np.sin(x[i]))

psi_bar = np.real(np.fft.fft2(psi)[0, 0]) / (M * N)
psi -= psi_bar

omega_hat = np.fft.fft2(omega)
psi_hat = np.zeros(shape = (M, N), dtype = 'complex')
k2 = np.zeros(shape = (M, N))
for i in range(0, M):
    for j in range(0, N):
        k2[i, j] = (-kx[i]**2 - ky[j]**2) if [i, j] != [0, 0] else np.inf
        psi_hat[i, j] = - omega_hat[i, j] / k2[i, j]

psi_recov = np.real(np.fft.ifft2(psi_hat))
e = psi_recov - psi

# Figure Setup Below:
psi_min, psi_max = np.floor(np.min(psi)), np.ceil(np.max(psi))
omega_min, omega_max = np.floor(np.min(omega)), np.ceil(np.max(omega))
print(np.min(e), np.max(e))

e_max = np.max(e)
e_order = np.floor(np.log10(e_max))
e_max = np.ceil(e_max * 10 ** -e_order) * 10 ** e_order
e_min = - e_max

fig, ax = plt.subplots(2, 2, constrained_layout = True)
fig.suptitle("Poisson Equation Solution Convergence\nComputational Grid Dimensions: {0}x{1}".format(M, N))

omega_plot = ax[0, 0].pcolor(x, y, np.transpose(omega), cmap = 'coolwarm', vmin = omega_min, vmax = omega_max)
ax[0, 0].set_aspect('equal')
ax[0, 0].set_title(r"Defined $\omega$")
fig.colorbar(omega_plot, ax = ax[0, 0])

psi_diff_plot = ax[0, 1].pcolor(x, y, np.transpose(e), cmap = 'coolwarm')#, vmin = e_min, vmax = e_max)
ax[0, 1].set_aspect('equal')
ax[0, 1].set_title(r"$\psi_{\mathrm{r}} - \psi_{\mathrm{a}}$")
fig.colorbar(psi_diff_plot, format = '%.0e', ax = ax[0, 1])

psi_plot = ax[1, 0].pcolor(x, y, np.transpose(psi), cmap = 'coolwarm', vmin = psi_min, vmax = psi_max)
ax[1, 0].set_aspect('equal')
ax[1, 0].set_title(r"Analytical $\psi_{\mathrm{a}}$")
fig.colorbar(psi_plot, ax = ax[1, 0])

psi_recov_con = ax[1, 1].pcolor(x, y, np.transpose(psi_recov), cmap = 'coolwarm', vmin = psi_min, vmax = psi_max)
ax[1, 1].set_aspect('equal')
ax[1, 1].set_title(r"Recovered $\psi_{\mathrm{r}}$")
fig.colorbar(psi_recov_con, ax = ax[1, 1])

for plt_i in [0, 1]:
    for plt_j in [0, 1]:
        ax[plt_i, plt_j].set_aspect('equal')

        ax[plt_i, plt_j].set_xlim(0, 2 * pi)
        ax[plt_i, plt_j].set_ylim(0, 2 * pi)

        ax[plt_i, plt_j].set_xlabel(r"$x$")
        ax[plt_i, plt_j].set_ylabel(r"$y$")


        ax[plt_i, plt_j].xaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
        ax[plt_i, plt_j].yaxis.set_major_formatter(FuncFormatter(lambda val,pos: '{:.0g}$\pi$'.format(val/pi) if val !=0 else '0'))
        ax[plt_i, plt_j].xaxis.set_major_locator(MultipleLocator(base = pi))
        ax[plt_i, plt_j].yaxis.set_major_locator(MultipleLocator(base = pi))

# fig.savefig("Poisson Solution {0}x{1}.png".format(M, N), dpi = 400)
plt.show()