import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 8, 100)
s = np.zeros(len(t))
for i in range(len(t)):
   if t[i] < 5:
      s[i] = np.sin(t[i])
   else:
      s[i] = 5 * np.sin(t[i])

approx = []
with open("approx.txt", "r+") as approx_txt:
   for n in approx_txt:
      approx.append(float(n))

approx = np.array(approx)

plt.plot(s)
plt.plot(approx)
plt.show()