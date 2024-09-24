import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def RK3(dydt, y0, t_eval, *args):
    """
    RK3 does the Runga-Kutta 3 Method on a differential system
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

def lorentz(t, u, c):
    # u = np.array([x, y, z])
    # c = np.array([beta, rho, sigma])

    return np.array([c[2] * (u[1] - u[0]), u[0] * (c[1] - u[2]) - u[1], u[0] * u[1] - c[0] * u[2]])

t = np.linspace(0, 40, 5000)
c1 = np.array([8/3, 26, 11])
c2 = np.array([8/3, 15, 25])
c3 = np.array([8/3, 15, 10])

u0 = np.array([1, 1, 1])

u1 = RK3(lorentz, u0, t, c1)
u2 = RK3(lorentz, u0, t, c2)
u3 = RK3(lorentz, u0, t, c3)

x1 = np.empty(len(t))
y1 = np.empty(len(t))
z1 = np.empty(len(t))

x2 = np.empty(len(t))
y2 = np.empty(len(t))
z2 = np.empty(len(t))

x3 = np.empty(len(t))
y3 = np.empty(len(t))
z3 = np.empty(len(t))

for i in range(len(t)):
    x1[i] = u1[i, 0]
    y1[i] = u1[i, 1]
    z1[i] = u1[i, 2]

    x2[i] = u2[i, 0]
    y2[i] = u2[i, 1]
    z2[i] = u2[i, 2]

    x3[i] = u3[i, 0]
    y3[i] = u3[i, 1]
    z3[i] = u3[i, 2]

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
# fig.suptitle(r"Lorentz System Animation")

def init():
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 100)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    return ax

def update(frame):
    f = int(frame)
    plt.cla()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 100)

    ax.plot(x1[:f], y1[:f], z1[:f], color = 'red', alpha = 0.2, label = r"$\vec{u}_1(0) = [1, 1, 1]$" + '\n' + r"$\beta = \frac{8}{3}$, $\rho = 26$, $\sigma = 11$")
    ax.scatter(x1[f], y1[f], z1[f], color = 'red')

    ax.plot(x2[:f], y2[:f], z2[:f], color = 'blue', alpha = 0.2, label = r"$\vec{u}_2(0) = [1, 1, 1]$" + '\n' + r"$\beta = \frac{8}{3}$, $\rho = 15$, $\sigma = 25$")
    ax.scatter(x2[f], y2[f], z2[f], color = 'blue')

    ax.plot(x3[:f], y3[:f], z3[:f], color = 'green', alpha = 0.2, label = r"$\vec{u}_3(0) = [1, 1, 1]$" + '\n' + r"$\beta = \frac{8}{3}$, $\rho = 15$, $\sigma = 10$")
    ax.scatter(x3[f], y3[f], z3[f], color = 'green')
    
    ax.set_title(r"Lorenz System, Governed by $\frac{d \vec{u}}{dt} = \left[ \sigma(y-x),\, x(\rho-z) - y, \, xy-\beta z\right]$")
    ax.text2D(0.5, 0.95, "Simulation & Animation done by Jacob Ivanov", horizontalalignment = "center", transform = ax.transAxes)
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.25), ncol = 3)
    ax.azim += 0.25
    print(f)
    return ax

ANI = ani.FuncAnimation(fig, update, frames = range(0, len(t)), init_func = init)
ANI.save("Lorentz System Trajectories.mp4", fps = 180, dpi = 200)