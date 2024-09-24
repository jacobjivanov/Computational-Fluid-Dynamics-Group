import ffti_v6 as fi
import numpy as np

# Hypothesis: `inter_1D()` itself must be flawed depending on input size
"""
Ni = 11
x = np.linspace(0, 2 * np.pi, Ni)

y = np.zeros(Ni)
for i in range(Ni):
    y[i] = np.e ** np.sin(x[i])

x_p = 1.3
a = fi.inter_1D(x, y, 1.3)
print(a)
print(np.e ** np.sin(x_p))
"""
# 

