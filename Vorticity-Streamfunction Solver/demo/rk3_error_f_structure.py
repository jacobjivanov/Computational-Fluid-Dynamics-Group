import numpy as np
import numba as nb

def y_ana(t, y0):
    return y0 * np.exp(1/2 * t ** 2)

@nb.njit()
def y_prime(t, y):
    return t * y

@nb.njit()
def mu(t):
    return np.exp(- 1/2 * t**2)

@nb.njit()
def muy_prime(t, y):
    return 0

a_RK3 = np.array([[8/15, 0], [1/4, 5/12]])
b_RK3 = np.array([1/4, 0, 3/4])
c_RK3 = np.array([0, 8/15, 2/3])

@nb.njit()
def rk_step(f, t0, y0, dt, a, b, c):
    s = c.size
    assert s > 0

    k = np.zeros(s)
    for rk_i in range(0, s):
        y_substep = y0
        for rk_j in range(0, rk_i):
            y_substep += a[rk_i - 1, rk_j] * k[rk_j] * dt
        t_substep = t0 + c[rk_i] * dt
        k[rk_i] = f(t_substep, y_substep)
    
    y_step = y0
    for rk_j in range(0, s):
        y_step += b[rk_j] * k[rk_j] * dt

    return y_step

T = 1000
t = np.linspace(0, 3, T)
dt = t[1]

y0 = 0.1
y1 = np.empty(T)
y1[0] = y0

y2 = np.empty(T)
y2[0] = y0

for n in range(1, T):
    y1[n] = rk_step(y_prime, t[n - 1], y1[n - 1], dt, a = a_RK3, b = b_RK3, c = c_RK3)
    y2[n] = rk_step(muy_prime, t[n - 1], y2[n - 1] * mu(t[n - 1]), dt, a = a_RK3, b = b_RK3, c = c_RK3) / mu(t[n])

import matplotlib.pyplot as plt
plt.figure(figsize=(7.5, 3.5), dpi = 200)
plt.semilogy(t, np.abs(y1 - y_ana(t, y0)), ':.', color = 'red', label = r"$ \frac{dy}{dt} = ty$")
plt.semilogy(t, np.abs(y2 - y_ana(t, y0)), ':.', color = 'blue', label = r"$\frac{d}{dt} \left[ \exp \left( - \frac{1}{2} t^2 \right) \cdot y \right] = 0$")
plt.xlabel("$t$")
plt.ylabel(r"$|y_{\mathrm{num}} - y_{\mathrm{ana}}|$")
plt.title("RK3 Integration Error with \nDifferent Time Derivative Structure")
plt.legend()
plt.savefig("RK3 Error, Time Derivative Structure.png", dpi = 200)