import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

def La_norm(e, dx, dy, a):
    N = len(e)
    La = 0
    for i in range(0, N):
        La += np.abs(e[i]) ** a
    La = (dx * dy * La) ** (1/a)
    return La

p = np.arange(5, 200)
L1_error = np.zeros(len(p))
L2_error = np.zeros(len(p))

for n in range(0, len(p)):
    x_dom, y_dom = [0, 2 * pi], [0, 2 * pi]
    L_x, L_y = x_dom[1] - x_dom[0], y_dom[1] - y_dom[0]
    M, N = p[n], p[n]
    dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N

    x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
    y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
    kx, ky = np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0)))

    omega = np.zeros(shape = (M, N))
    psi = np.zeros(shape = (M, N))

    for i in range(0, M):
        for j in range(0, N):
            # Analytical Solution, "Fourier Simple"
            # psi[i, j] = np.sin(x[i]) * np.cos(y[j])
            # omega[i, j] = 2 * psi[i, j]
            
            # Analytical Solution, Not "Fourier Simple"
            psi[i, j] = np.exp(np.sin(x[i]) + np.cos(y[j]))
            omega[i, j] = - psi[i, j] * (np.cos(x[i]) ** 2 - np.cos(y[j]) - np.sin(x[i]) + np.sin(y[j]) ** 2)

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

    L1_error[n] = La_norm(np.ravel(e), dx, dy, 1)
    L2_error[n] = La_norm(np.ravel(e), dx, dy, 2)
    print(p[n])

plt.loglog(p, L1_error)
plt.loglog(p, L2_error)
plt.show()
