import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def rk_step(f, y0, t0, dt, a, b, c):
    s = c.size

    k = np.zeros(s)
    for i in range(0, s):
        y_substep = y0
        for j in range(0, i):
            y_substep += a[i - 1, j] * k[j] * dt
        k[i] = f(t0 + c[i] * dt, y_substep)
    
    y_step = y0
    for j in range(0, s):
        y_step += b[j] * k[j] * dt

    return y_step

# RK4

a = np.array([[1/2, 0, 0], 
              [0, 1/2, 0], 
              [0, 0, 1]])
b = np.array([1/6, 1/3, 1/3, 1/6])
c = np.array([0, 1/2, 1/2, 1])

"""
a = np.array([[8/15, 0], [1/4, 5/12]])
b = np.array([1/4, 0, 3/4])
c = np.array([0, 8/15, 2/3])
"""

def La_Norm(e, dx, a):
    N = len(e)
    La = 0
    for i in range(0, N):
        La += np.abs(e[i]) ** a
    La = (dx * La) ** (1/a)
    return La

def dudt(t, u):
    # return 1/u
    # return u
    return np.sqrt(u)

G = np.arange(5, 100) ** 2
l1_error = np.zeros(len(G))
l2_error = np.zeros(len(G))

for g in range(0, G.size):
    t_dom = [0, 10]
    T = G[g]
    t = np.linspace(t_dom[0], t_dom[1], T, endpoint = False)
    dt = t_dom[1]/T

    u_num = np.empty(shape = (T))
    u_ana = np.empty(shape = (T))
    u_num[0] = 2
    u_ana[0] = 2

    for n in range(0, T - 1):
        u_num[n + 1] = rk_step(dudt, u_num[n], t[n], dt, a = a, b = b, c = c)
        
        # u_ana[n + 1] = np.sqrt(2 * t[n + 1] + 4)
        # u_ana[n + 1] = u_ana[0] * np.exp(t[n + 1])
        u_ana[n + 1] = (t[n + 1] + np.sqrt(8)) ** 2 / 4

    e = u_num - u_ana
    
    l1_error[g] = La_Norm(e, dt, 1)
    l2_error[g] = La_Norm(e, dt, 2)


plt.loglog(G, l1_error, ':o', label = "L1 Error", alpha = 0.5)
plt.loglog(G, l2_error, ':o', label = "L2 Error", alpha = 0.5)

plt.loglog(G, 1 / (G ** 1), linestyle = 'dashed', label = '1st Order Convergence', color = 'grey')
plt.loglog(G, 1 / (G ** 2), linestyle = 'dashed', label = '2nd Order Convergence', color = 'grey')
plt.loglog(G, 1 / (G ** 3), linestyle = 'dashed', label = '3rd Order Convergence', color = 'grey')
plt.loglog(G, 1 / (G ** 4), linestyle = 'dashed', label = '4th Order Convergence', color = 'grey')
plt.ylabel("Error")
plt.xlabel(r"$N$")
plt.legend()

# plt.title(r"RK3 Error Convergence for $\frac{du}{dt} = \frac{1}{u}, u(t = 0) = 2 $")
# plt.title(r"RK3 Error Convergence for $\frac{du}{dt} = u, u(t = 0) = 2 $")
plt.title(r"RK4 Error Convergence for $\frac{du}{dt} = \sqrt{u}, u(t = 0) = 2 $")

plt.show()


"""
plt.plot(t, np.log(u_num) - np.log(u_ana))
# plt.plot(t, )
plt.show()
"""