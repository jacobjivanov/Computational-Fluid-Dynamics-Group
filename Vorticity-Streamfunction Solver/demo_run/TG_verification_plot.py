import numpy as np
from numpy.fft import fft2, ifft2
from numpy import real
import matplotlib.pyplot as plt

def l2_norm(error):
    l2 = (4*np.pi*np.pi/65536 * np.sum(np.abs(error) ** 2)) ** (1/2)
    return l2

nu = 9e-4
x = np.linspace(0, 2*np.pi, 256, endpoint = False)
y = np.linspace(0, 2*np.pi, 256, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')

t000 = np.empty(2400)
t002 = np.empty(2400)
e000 = np.empty(2400)
e002 = np.empty(2400)

for i in range(0, 2400):
    data_000 = np.load("TG_verification/256 0/s = {0}.npz".format(i*5))
    data_002 = np.load("TG_verification/256 0.02/s = {0}.npz".format(i*5))
    
    t000_now = data_000['t']
    t002_now = data_002['t']
    t000[i] = t000_now
    t002[i] = t002_now

    omega000_now = real(ifft2(data_000['Omega']))
    omega002_now = real(ifft2(data_002['Omega']))

    omega_TG = -2*np.exp(-2*nu*t000_now) * np.cos(x_grid) * np.cos(y_grid)
    e000[i] = l2_norm(omega_TG - omega000_now)

    omega_TG = -2*np.exp(-2*nu*t002_now) * np.cos(x_grid) * np.cos(y_grid)
    e002[i] = l2_norm(omega_TG - omega002_now)
    print(t002_now, e002[i])

plt.figure(figsize = (7.5, 3.75), dpi = 200)
plt.title("Error of Numerical and Analytical Taylor-Green Vorticity")
plt.xlabel("$t$")
plt.ylabel(r"$\ell_2|\omega_\mathrm{num} - \omega_\mathrm{TG}|$")
plt.semilogy(t000, e000, color = 'red', label = r"$|\gamma| = 0.00$")
plt.semilogy(t002, e002, color = 'blue', label = r"$|\gamma| = 0.02$")
plt.legend()
plt.savefig("TG_verification_plot.png", dpi = 200)
plt.show()
