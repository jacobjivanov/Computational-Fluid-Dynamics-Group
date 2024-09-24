import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

def RK3_step(pypt, y0, dt, *args): # version 11/03/2023
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
        orig_dtype = y0.dtype

        y0 = y0.flatten()
        N = y0.size

        pypt_0 = pypt(y0.reshape(orig_shape)).flatten()
        y_step1 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step1[i] = y0[i] + dt * 8/15 * pypt_0[i]

        pypt_step1 = pypt(y_step1.reshape(orig_shape)).flatten()
        y_step2 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step2[i] = y_step1[i] + dt * (-17/60 * pypt_0[i] + 5/12 * pypt_step1[i])

        pypt_step2 = pypt(y_step2.reshape(orig_shape)).flatten()
        y_step3 = np.empty(N, dtype = orig_dtype)
        for i in range(0, N): y_step3[i] = y_step2[i] + dt * (-5/12 * pypt_step1[i] + 3/4 * pypt_step2[i])
    
        return y_step3.reshape(orig_shape)

# computational grid parameters
M = 512
x_dom = [0, 2 * pi]
x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
dx = x[1] - x[0]
kx = wavenumbers(M, x_dom)

nu = 1e-2 # kinematic viscosity

t, t_end = 0, 10

u = np.empty(M)
for i in range(0, M):
    u[i] = np.exp(np.sin(x[i]))
U = np.fft.fft(u)

def dUdt(U):
    # \pdv{u}{t} = \nu \pdv[2]{u}{x} - \pdv{u}{x}
    dUdt = np.empty(M, dtype = 'complex')
    for p in range(0, M):
        dUdt[p] = - (nu * kx[p] - 1j) * kx[p] * U[p]
    return dUdt

n = 0
while t < t_end:
    if n % 100 == 0:
        plt.plot(x, np.real(np.fft.ifft(U)))
        plt.show()
    vis_dt = (dx**2)/nu
    adv_dt = dx/1

    dt = 0.5 * min(vis_dt, adv_dt)
    U = RK3_step(dUdt, U, dt)
    t += dt
    n += 1