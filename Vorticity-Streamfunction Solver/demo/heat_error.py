import numpy as np
from numpy.fft import fft, ifft
from numpy import real, imag
import matplotlib.pyplot as plt

def l2_norm(error, dx):
    l2 = (dx * np.sum(np.abs(error) ** 2)) ** (1/2)
    return l2

def rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + 8/15*dt, y0 + dt*8/15*k1)
    k3 = f(t0 + 2/3*dt, y0 + dt * (k1/4 + 5*k2/12))

    y3 = y0 + dt * (k1/4 + 3*k3/4)
    return y3

def dUdt(t, U):
    return -nu*kp*kp*U

def theta_ana(t):
    return -np.exp(-nu*16*t)*np.sin(4*x) + 10*np.exp(-nu*100*t)*np.sin(10*x)

M = np.array([2*n for n in range(10, 200)])
l2 = np.empty(len(M))
for m in range(0, len(M)):
    x = np.linspace(0, 2*np.pi, M[m], endpoint = False)
    theta = -1*np.sin(4*x) + 10*np.sin(10*x)
    nu = 1
    dx = x[1]

    kp = np.array(list(range(0, M[m]//2 + 1)) + list(range(- M[m]//2 + 1, 0)))

    Theta = fft(theta)

    t = 0
    dt = nu * dx**2/4
    while t < 1:
        Theta = rk3_step(dUdt, t, Theta, dt)
        theta = real(ifft(Theta))
        t += dt
    l2[m] = l2_norm(theta - theta_ana(t), dx)

plt.figure(figsize = (7.5, 3.5), dpi = 200)
plt.ylabel(r"$\ell_2(\mathbf{e})$")
plt.xlabel("M")
plt.title("Convergence of Heat Equation Solution at $t = 1$")
plt.loglog(M, l2, ':.', color = 'blue')
plt.savefig("Heat Error.png", dpi = 200)