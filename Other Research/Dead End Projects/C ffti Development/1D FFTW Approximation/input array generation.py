import numpy as np

# t = np.linspace(0, 35, 100)
# s = np.log(t + 1) * (15 - np.sin(t))

t = np.linspace(0, 8, 15)
s = np.zeros(len(t))
for i in range(len(t)):
   # if t[i] < 5:
   #    s[i] = np.sin(t[i])
   # else:
   #    s[i] = 5 * np.sin(t[i])
   s[i] = np.sin(t[i])


# -----------------------
t_string = '{'
s_string = '{'

for i in range(len(t)):
   t_string += str(t[i]) + ','
   s_string += str(s[i]) + ','

t_string += '};'
s_string += '};'

print(t_string)
print(s_string)