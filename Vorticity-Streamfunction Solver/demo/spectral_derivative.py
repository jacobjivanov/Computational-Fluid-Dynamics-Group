# The following script was written by Jacob Ivanov, Undergraduate Researcher for the Computational Fluid Dynamics Group at the University of Connecticut, under Dr. Georgios Matheou. 

import numpy as np

M = np.array([i for i in range(5, 100)] + [i for i in range(100, 1000, 10)])
error = np.empty(len(M))

for i in range(0, len(M)):
    m = M[i]
    x = np.linspace(0, 2*np.pi, m, endpoint = False)
    y = np.exp(np.cos(x)) * np.sin(x)

    kp = np.array(list(range(0, m//2 + 1)) + list(range(- m//2 + 1, 0)))

    Y = np.fft.fft(y)
    Y_prime = 1j * kp * Y
    y_prime_num = np.real(np.fft.ifft(Y_prime))
    y_prime_ana = - np.exp(np.cos(x)) * (np.sin(x)**2 - np.cos(x))
    
    error[i] = np.max(np.abs(y_prime_num - y_prime_ana))

import matplotlib.pyplot as plt
plt.figure(figsize = (7.5, 3.5), dpi = 200)
plt.loglog(M, error, ':.', color = 'blue')
plt.axhline(2**-53, color = 'grey', linestyle = 'dashed', label = "Double Precision")
plt.title("Figure 1: Convergence of Spectral and Analytical Derivative")
plt.ylabel(r"$\max \left| y'_\mathrm{num} - y'_\mathrm{ana} \right|$")
plt.legend()
plt.xlabel(r"$M$")
plt.savefig("spectral_derivative.png", dpi = 200)