import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def f(x, y):
    # return np.exp(np.sin(x) + np.sin(y))# + np.sin(x * y)
    if abs(x) < 1 and abs(y) < 1:
    # if 1 < (x ** 2 + y ** 2) < np.sqrt(2):
        return 1
    else:
        return 0.01
    
# initialization
x_dom = [- np.pi, np.pi]
y_dom = [- np.pi, np.pi]

Ni, Nj = 50, 50

x = np.linspace(x_dom[0], x_dom[1], Ni + 1)[:-1]
y = np.linspace(y_dom[0], y_dom[1], Nj + 1)[:-1]
xy = np.meshgrid(x, y, indexing = 'ij')

u = np.empty(shape = (Ni, Nj))

for i in range(Ni):
    for j in range(Nj):
        u[i, j] = f(x[i], y[j])

X = np.fft.fftfreq(x.size)
Y = np.fft.fftfreq(y.size)
XY = np.meshgrid(X, Y, indexing = 'ij')

U = np.fft.fft2(u, norm = 'ortho')
print(U.shape)
U_mag = np.empty(shape = U.shape)
U_phase = np.empty(shape = U.shape)

for i in range(Ni):
    for j in range(Nj):
        U_mag[i, j] = abs(U[i, j])
        U_phase[i, j] = np.arctan2(np.imag(U[i, j]), np.real(U[i, j]))

fig1 = plt.figure()
fig1.suptitle("Physical Space")
ax1 = fig1.add_subplot(projection = '3d')
img1 = ax1.scatter(xy[0], xy[1], u, c = u, cmap = 'plasma', alpha = 0.5)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$u$')
plt.show(block = False)

fig2 = plt.figure()
fig2.suptitle("Frequency Space (Magnitude)")
ax2 = fig2.add_subplot(projection = '3d')
img2 = ax2.scatter(XY[0], XY[1], U_mag, c = np.log10(U_mag + 1), cmap = 'plasma', alpha = 0.5)
ax2.set_zlim(0, 10)
ax2.set_xlabel(r'$\hat{x}$')
ax2.set_ylabel(r'$\hat{y}$')
ax2.set_zlabel(r'$\log_{10} \left[ ||\hat{u}|| + 1 \right]$')
plt.show(block = False)

fig3 = plt.figure()
fig3.suptitle("Frequency Space (Phase)")
ax3 = fig3.add_subplot(projection = '3d')
img3 = ax3.scatter(XY[0], XY[1], U_phase, c = U_phase, cmap = 'plasma', alpha = 0.5)
ax3.set_xlabel(r'$\hat{x}$')
ax3.set_ylabel(r'$\hat{y}$')
ax3.set_zlabel(r'$\arctan2 \left[ \frac{\Im(\hat{u})}{\Re(\hat{u})} \right]$')
plt.show()