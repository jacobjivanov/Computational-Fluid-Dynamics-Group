# The following script was written by Jacob Ivanov, Undergraduate Researcher for the Computational Fluid Dynamics Group at the University of Connecticut, under Dr. Georgios Matheou. 

import numpy as np
from numpy.fft import fft, ifft
from numpy import real, imag

import matplotlib.pyplot as plt
sm = plt.cm.ScalarMappable(cmap = 'coolwarm')
sm.set_clim(0, 10)
plt.figure(figsize = (7.5, 3.5), dpi = 200)

M = 64
x = np.linspace(0, 2*np.pi, M, endpoint = False)
dx = x[1]
u = np.exp(np.sin(x))
U_a = fft(u)
U_c = fft(u)

kp = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

def rk3_step(f, t0, y0, dt):
    k1 = f(t0, y0)
    k2 = f(t0 + 8/15*dt, y0 + dt*8/15*k1)
    k3 = f(t0 + 2/3*dt, y0 + dt * (k1/4 + 5*k2/12))

    y3 = y0 + dt * (k1/4 + 3*k3/4)
    return y3

def dUdt_advective(t, U):
    u = real(ifft(U))
    dudx = real(ifft(1j*kp*U))
    return -fft(u*dudx)

def dUdt_convective(t, U):
    u = real(ifft(U))
    u2 = u**2
    dU2dx = 1j*kp*fft(u2)
    return -1/2 * dU2dx

t = 0
dt = dx / 2
n = 0
while n < 11:
    U_a = rk3_step(dUdt_advective, t, U_a, dt,)
    # U_c = rk3_step(dUdt_convective, t, U_c, dt)

    if n in [0, 5, 8, 9, 10]:
        u_a = real(ifft(U_a))
        # u_c = real(ifft(U_c))
        
        color = sm.to_rgba(n)
        plt.plot(x, u_a, color = color, label = "timestep: {0}".format(n))
        # plt.plot(x, u_c, label = 'Convective')
        
    n += 1
    t += dt

plt.title("Burger's Equation Evolution")
plt.xlabel("$x$")
plt.ylabel("$u$")
plt.legend()
plt.savefig("Burger's Equation Evolution.png", dpi = 200)