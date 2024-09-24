import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as ani
# import ffti_v10 as fi
from time import time

def RK3a2_step(pypt, y0, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)[i, j]  
        y0: current array of y[i, j]
        dt: time step size
    """

    M, N = y0.shape

    y_step1 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_0 = pypt(y0)
    for i in range(0, M):
        for j in range(0, N):
            y_step1[i, j] = y0[i, j] + dt * 8/15 * pypt_0[i, j]

    y_step2 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_step1 = pypt(y_step1)
    for i in range(0, M):
        for j in range(0, N):
            y_step2[i, j] = y_step1[i, j] + dt * (-17/60 * pypt_0[i, j] + 5/12 * pypt_step1[i, j])

    y_step3 = np.empty(shape = (M, N), dtype = 'complex')
    pypt_step2 = pypt(y_step2)
    for i in range(0, M):
        for j in range(0, N):
            y_step3[i, j] = y_step2[i, j] + dt * (-5/12 * pypt_step1[i, j] + 3/4 * pypt_step2[i, j])
    
    return y_step3

def RK3v_step(pypt, y0, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)[i]
        y0: current array of y[i]
        dt: time step size
    """

    M, = y0.shape

    y_step1 = np.empty(shape = (M, 1), dtype = 'complex')
    pypt_0 = pypt(y0)
    for i in range(0, M):
        y_step1[i] = y0[i] + dt * 8/15 * pypt_0[i]

    y_step2 = np.empty(shape = (M, 1), dtype = 'complex')
    pypt_step1 = pypt(y_step1)
    for i in range(0, M):
        y_step2[i] = y_step1[i] + dt * (-17/60 * pypt_0[i] + 5/12 * pypt_step1[i])

    y_step3 = np.empty(shape = (M, 1), dtype = 'complex')
    pypt_step2 = pypt(y_step2)
    for i in range(0, M):
        y_step3[i] = y_step2[i] + dt * (-5/12 * pypt_step1[i] + 3/4 * pypt_step2[i])
    
    return y_step3

def RK3s_step(pypt, y0, dt):
    """
    RK3_step does the Runga-Kutta 3 Method on a differential system for a single time step
    Inputs:
        pypt: current array of (∂y/∂t)
        y0: current array of y[i]
        dt: time step size
    """

    pypt_0 = pypt(y0)
    y_step1 = y0 + dt * 8/15 * pypt_0

    pypt_step1 = pypt(y_step1)
    y_step2 = y_step1 + dt * (-17/60 * pypt_0 + 5/12 * pypt_step1)

    pypt_step2 = pypt(y_step2)
    y_step3 = y_step2 + dt * (-5/12 * pypt_step1 + 3/4 * pypt_step2)
    
    return y_step3

def dydt(y):
    return 1/y

t_dom = [0, 10]
T = 1000
t = np.linspace(t_dom[0], t_dom[1], T, endpoint = False)
dt = t_dom[1]/T

y = np.empty(shape = (T))
y[0] = 2

for n in range(0, T - 1):
    y[n + 1] = RK3s_step(dydt, y[n], dt)

plt.plot(t, y)
plt.show()