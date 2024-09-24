import numpy as np

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
    
    else:
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