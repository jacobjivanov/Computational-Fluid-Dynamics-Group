import numpy as np

def RK3(dydt, y0, t_eval, *args):
    """
    RK3 does the Runga-Kutta 3 Method on a differential system. Written entirely by Jacob Ivanov
    Inputs:
        dydt: callable function of form y'(t, y, *args)
        y0: initial array of y, i.e. y(t = 0)
        t_eval: array of times at which to compute the value of y
    """
    
    y = np.empty(shape = (len(t_eval))) if y0.shape[0] == 1 else np.empty(shape = (len(t_eval), y0.shape[0]))
    y[0] = y0

    for t in range(0, len(t_eval) - 1):
        dt = t_eval[t + 1] - t_eval[t]
        
        y_na1 = y[t] + dt * 8/15 * dydt(t_eval[t], y[t], *args)
        y_na2 = y_na1 + dt * (-17/60 * dydt(t_eval[t], y[t], *args) + 5/12 * dydt(t_eval[t] + dt/3, y_na1, *args))
        y[t + 1] = y_na2 + dt * (-5/12 * dydt(t_eval[t] + dt/3, y_na1, *args) + 3/4 * dydt(t_eval[t] + 2 * dt/3, y_na2, *args))

    return y

def La_Norm(e, dx, a):
    N = len(e)
    La = 0
    for i in range(0, N):
        La += np.abs(e[i]) ** a
    La = (dx * La) ** (1/a)
    return La

def dydt(t, y):
    return y

y0 = np.array([1])

N = np.arange(5, 100) ** 2 # change to 3 for roundoff stuff, but that takes a minute or two to run
l1_error = np.zeros(len(N))
l2_error = np.zeros(len(N))

for n in range(len(N)):
    t = np.linspace(0, 2, int(N[n]))
    e = RK3(dydt, y0, t) - np.exp(t)

    """
    for i in range(N[n]):
        l1_error[n] += np.abs(e[i])
        l2_error[n] += np.abs(e[i]) ** 2
        
    l1_error[n] = l1_error[n]
    l2_error[n] *= t[1]
    l2_error[n] = np.sqrt(l2_error[n])
    """
    
    l1_error[n] = La_Norm(e, t[1], 1)
    l2_error[n] = La_Norm(e, t[1], 2)

import matplotlib.pyplot as plt
# plt.rc('text', usetex = False)
# plt.rc('text.latex', preamble = r'\usepackage{physics} \usepackage{DejaVuSans}')
plt.loglog(N, l1_error, ':o', label = "L1 Error", alpha = 0.5)
plt.loglog(N, l2_error, ':o', label = "L2 Error", alpha = 0.5)
plt.ylabel("Absolute Error")
plt.xlabel(r"$N$")
plt.legend()
plt.title(r"RK3 Error Convergence for $\frac{dy}{dt} = y, y(t = 0) = 1 $")
# plt.title(r"RK3 Error Convergence for $\dv{y}{t} = y, y(t = 0) = 1 $")
# plt.savefig("RK3 Convergence v4.png", dpi = 200)
plt.show()