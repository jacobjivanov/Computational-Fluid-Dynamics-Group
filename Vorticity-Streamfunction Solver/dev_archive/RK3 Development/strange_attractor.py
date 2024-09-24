import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def RK3_step(pypt, y0, dt, *args):
    """
    Performs the specialized 3rd-Order Runga Kutta method defined in "Spectral Methods for the Navier-Stokes Equations..." by Spalart, Moser, Rogers. Is able to process inputs of arbitrary dimensions, including scalar, vector, and larger arrays by flattening to a vector and later rebuilding the original dimensions. This function was last updated 10/28/2023 by JJI. 
    
    Function Inputs:
        pypt: time derivative of state y0 of form pypt(y0, *args). 
              must be autonomous (time-independent)
        y0: current state
        dt: time step
        *args: other pypt function arguments
    """

    y0 = np.asarray(y0)

    if y0.ndim == 0: # if scalar
        
        pypt_0 = pypt(y0)
        y_step1 = y0 + dt * 8/15 * pypt_0

        pypt_step1 = pypt(y_step1)
        y_step2 = y_step1 + dt * (-17/60 * pypt_0 + 5/12 * pypt_step1)

        pypt_step2 = pypt(y_step2)
        y_step3 = y_step2 + dt * (-5/12 * pypt_step1 + 3/4 * pypt_step2)
        
        return y_step3
    
    else: # if vector/array
        orig_shape = y0.shape
        y0.flatten()
        N = y0.size

        pypt_0 = pypt(y0.reshape(orig_shape)).flatten()
        y_step1 = np.empty(N)
        for i in range(0, N): y_step1[i] = y0[i] + dt * 8/15 * pypt_0[i]

        pypt_step1 = pypt(y_step1.reshape(orig_shape)).flatten()
        y_step2 = np.empty(N)
        for i in range(0, N): y_step2[i] = y_step1[i] + dt * (-17/60 * pypt_0[i] + 5/12 * pypt_step1[i])

        pypt_step2 = pypt(y_step2.reshape(orig_shape)).flatten()
        y_step3 = np.empty(N)
        for i in range(0, N): y_step3[i] = y_step2[i] + dt * (-5/12 * pypt_step1[i] + 3/4 * pypt_step2[i])

        return y_step3.reshape(orig_shape)

def four_wing(u):
    # u = np.array([x, y, z])
    [a, b, c] = [0.2, 0.01, -0.4]
    u_t = np.array([
        a * u[0] + u[1] * u[2], 
        b * u[0] + c * u[1] - u[0] * u[2],
        -u[2] - u[0] *  u[1]
        ])

    print(u_t)
    return u_t

def lorenz(u):
    [sigma, rho, beta] = [10, 28, 8/3]

    u_t = np.array([
        sigma * (-u[0] + u[1]), 
        - u[0] * u[2] + rho * u[0] - u[1],
        u[0] * u[1] - beta * u[2] 
    ])
    
    return u_t

# u = np.empty(N)
u = np.array([1.3, -0.18, 0.01])

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
    
    return ax

def update(frame):
    for t in range(0, 10):
        dt = 0.01
        global u 
        # u = [u[0] + 1, u[1] + 1, u[2] + 1]
        u = RK3_step(four_wing, u, dt)

    plt.cla()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(0, 100)

    ax.scatter(u[0], u[1], u[2], color = 'red')

    return ax

anim = animation.FuncAnimation(fig, update, frames = np.arange(0, 1000),  init_func = init, interval = 50)
plt.show()