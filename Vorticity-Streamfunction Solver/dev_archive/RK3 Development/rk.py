import numpy as np
import numba as nb
import matplotlib; print(matplotlib.get_backend())

a_RK4 = np.array([[1/2, 0, 0], 
              [0, 1/2, 0], 
              [0, 0, 1]])
b_RK4 = np.array([1/6, 1/3, 1/3, 1/6])
c_RK4 = np.array([0, 1/2, 1/2, 1])

a_RK3 = np.array([[8/15, 0], [1/4, 5/12]])
b_RK3 = np.array([1/4, 0, 3/4])
c_RK3 = np.array([0, 8/15, 2/3])

# @nb.njit()
def rk_step_s(f, y0, t0, dt, a = a_RK3, b = b_RK3, c = c_RK3):
    s = c.size

    k = np.zeros(s)
    for rk_i in range(0, s):
        y_substep = y0
        for rk_j in range(0, rk_i):
            y_substep += a[rk_i - 1, rk_j] * k[rk_j] * dt
        k[rk_i] = f(t0 + c[rk_i] * dt, y_substep)
    
    y_step = y0
    for rk_j in range(0, s):
        y_step += b[rk_j] * k[rk_j] * dt

    return y_step

# @nb.njit()
def rk_step(f, t0, y0, dt, a = a_RK3, b = b_RK3, c = c_RK3):
    s = c.size

    k = np.zeros(shape = (s, *y0.shape), dtype = 'complex')
    for rk_i in range(0, s):
        y_substep = y0
        for rk_j in range(0, rk_i):
            y_substep += a[rk_i - 1, rk_j] * k[rk_j] * dt
        t_substep = t0 + c[rk_i] * dt
        k[rk_i::] = f(t_substep, y_substep)
    
    y_step = y0
    for rk_j in range(0, s):
        y_step += b[rk_j] * k[rk_j] * dt

    return y_step