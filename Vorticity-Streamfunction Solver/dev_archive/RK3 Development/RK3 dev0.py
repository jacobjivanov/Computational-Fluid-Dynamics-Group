import numpy as np

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