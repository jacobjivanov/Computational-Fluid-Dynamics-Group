import numpy as np
import matplotlib.pyplot as plt

N = 50

x = np.linspace(0, 2 * np.pi, N)
x_string = "{"
for i in x:
   x_string += "{0:.4f}".format(i)
   x_string += ", "
   # print("{0:.4f}".format(i))
x_string += "}"

y = np.zeros(N)
y_string = "{"
for i in range(len(x)):
   # print(i)
   if x[i] < 3:
      y[i] = np.log(x[i] + 1)
   else:
      y[i] = np.log(x[i] + 1) + 1

   y_string += "{0:.4f}".format(y[i])
   y_string += ", "
   # print("{0:.4f}".format(y[i]))
y_string += "}"

print(x_string)
print()
print(y_string)

plt.plot(x, y)
plt.show()