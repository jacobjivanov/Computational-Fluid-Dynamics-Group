# The following script was written by Jacob Ivanov, Undergraduate Researcher for the Computational Fluid Dynamics Group at the University of Connecticut, under Dr. Georgios Matheou. 

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize = (7.5, 4.5), dpi = 200)

M = 256
x = np.linspace(0, 2*np.pi, M, endpoint = False)
dx = x[1]

kp = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

def l2_norm(error, dx):
    l2 = (dx * np.sum(np.abs(error) ** 2)) ** (1/2)
    return l2

def euler_rk1_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    y1 = y0 + dt*k1

    return y1

def ralston_rk2_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + 2*dt/3, y0 + 2*dt*k1/3)

    y2 = y0 + dt * (k1/4 + 3*k2/4)
    return y2

def heun_rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + dt/3, y0 + dt*k1/3)
    k3 = f(t0 + 2*dt/3, y0 + 2*dt*k2/3)

    y3 = y0 + dt * (k1/4 + 3*k3/4)
    return y3

def wray_rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + 8/15*dt, y0 + dt*8/15*k1)
    k3 = f(t0 + 2/3*dt, y0 + dt * (k1/4 + 5*k2/12))

    y3 = y0 + dt * (k1/4 + 3*k3/4)
    return y3

def sspr_rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + dt, y0 + dt*k1)
    k3 = f(t0 + dt/2, y0 + dt * (k1/4 + k2/4))

    y3 = y0 + dt * (k1/6 + k2/6 + 2*k3/3)
    return y3

def ralston_rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + dt/2, y0 + dt*k1/2)
    k3 = f(t0 + 3*dt/4, y0 + 3*dt*k2/4)

    y3 = y0 + dt * (2*k1/9 + k2/3 + 4*k3/9)
    return y3

def classic_rk4_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + dt/2, y0 + dt*k1/2)
    k3 = f(t0 + dt/2, y0 + dt*k2/2)
    k4 = f(t0 + dt, y0 + dt*k3)

    y4 = y0 + dt * (k1/6 + k2/3 + k3/3 + k4/6)
    return y4

def pThetapt(t, Theta):
    pThetapx = 1j * kp * Theta
    return - pThetapx

method = [euler_rk1_step, ralston_rk2_step, heun_rk3_step, wray_rk3_step, sspr_rk3_step, ralston_rk3_step, classic_rk4_step]

method_name = [
    "Euler's RK1",
    "Ralston's RK2",
    "Heun's RK3",
    "Wray's RK3",
    "SSPR RK3",
    "Ralston's RK3",
    "Classic RK4"
]

for m in range(0, len(method)):
    cfl_max = np.sort(np.random.random(100))
    l2 = np.empty(len(cfl_max))
    for i in range(0, len(cfl_max)):
        theta = np.exp(np.cos(x))
        Theta = np.fft.fft(theta)
        Theta_mag0 = np.sum(np.abs(Theta) ** 2)
        t = 0
        n = 0
        dt = cfl_max[i] * dx
        while t < 10 and np.sum(np.abs(Theta) ** 2) < 10 * Theta_mag0:
            Theta = method[m](pThetapt, t, Theta, dt)
            t += dt
            n += 1

        if np.sum(np.abs(Theta) ** 2) > 10 * Theta_mag0:
            l2[i] = 10
            continue
        
        theta = np.real(np.fft.ifft(Theta))
        theta_ana = np.exp(np.cos(x - t))
        l2[i] = l2_norm(theta - theta_ana, dx)

    plt.loglog(cfl_max, l2, ':.', label = method_name[m])
    print(method_name[m], " complete")

plt.ylim(1e-14, 1)
plt.title(r"Convergence of $\theta(t = 10)$ Solutions for Decreasing $CFL_\mathrm{max}$")
plt.ylabel(r"$\ell_2 [\theta_\mathrm{num} - \theta_\mathrm{ana}]$")
plt.xlabel(r"$CFL_\mathrm{max}$")
plt.legend()
plt.savefig("advection_error.png", dpi = 200)