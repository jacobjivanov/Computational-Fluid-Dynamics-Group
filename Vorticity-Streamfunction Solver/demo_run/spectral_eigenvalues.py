# The following script was written by Jacob Ivanov, Undergraduate Researcher for the Computational Fluid Dynamics Group at the University of Connecticut, under Dr. Georgios Matheou. 

import numpy as np
from numpy.fft import fft, ifft
from numpy import real, imag
import matplotlib.pyplot as plt

# https://www.desmos.com/calculator/e8arccoku0
plt.figure(figsize = (7.5, 3.5), dpi = 200)
for m in np.array([10, 50, 100, 250]):
    D = np.empty(shape = (m, m))
    for i in range(0, m):
        for j in range(0, m):
            if i == j: 
                D[i, j] = 0
            if i != j:
                a = np.pi * (i - j)
                b = a/m
                D[i, j] = (np.cos(a)/np.tan(b) - 1/m * np.sin(a)/np.sin(b)**2)/2

    eig_real = real(np.linalg.eig(D)[0])
    eig_imag = imag(np.linalg.eig(D)[0])

    plt.scatter(eig_real/m, eig_imag/m, label = "M = {0}".format(m), alpha = 0.5)

plt.xlabel(r"$\Re[ \lambda ]/M$")
plt.ylabel(r"$\Im[ \lambda ]/M$")
plt.legend()
plt.title("Spectral Advection Eigenvalues")
plt.savefig("Spectral Eigenvalues.png", dpi = 200)