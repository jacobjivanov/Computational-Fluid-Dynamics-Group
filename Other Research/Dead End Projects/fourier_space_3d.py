import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def f(x, y, z):    
    return np.abs(np.sin(x) + np.sin(y * z)) + 1
    if abs(x) < 1 and abs(y) < 1:#and abs(z) < 1:
        return 10
    else:
        return 1

# initialization
x_dom = [- np.pi, np.pi]
y_dom = [- np.pi, np.pi]
z_dom = [- np.pi, np.pi]

Ni, Nj, Nk = 20, 20, 20

x = np.linspace(x_dom[0], x_dom[1], Ni + 1)[:-1]
y = np.linspace(y_dom[0], y_dom[1], Nj + 1)[:-1]
z = np.linspace(z_dom[0], z_dom[1], Nk + 1)[:-1]
xyz = np.meshgrid(x, y, z, indexing = 'ij')

u = np.empty(shape = (Ni, Nj, Nk))

for i in range(Ni):
    for j in range(Nj):
        for k in range(Nk):
            u[i, j, k] = f(x[i], y[j], z[k])

X = np.fft.fftfreq(x.size)
Y = np.fft.fftfreq(y.size)
Z = np.fft.fftfreq(z.size)
XYZ = np.meshgrid(X, Y, Z, indexing = 'ij')

U = np.fft.fftn(u)
U_mag = np.empty(shape = U.shape)
U_phase = np.empty(shape = U.shape)

for i in range(Ni):
    for j in range(Nj):
        for k in range(Nk):
            U_mag[i, j, k] = abs(U[i, j, k])
            U_phase[i, j] = np.arctan2(np.imag(U[i, j]), np.real(U[i, j]))


fig1 = plt.figure()
fig1.suptitle("Physical Space")
ax1 = fig1.add_subplot(projection = '3d')
img1 = ax1.scatter(xyz[0], xyz[1], xyz[2], c = u, cmap = 'Blues', alpha = 0.1, norm = colors.LogNorm(np.amin(u), np.amax(u)))
fig1.colorbar(img1, location = 'left', label = r'$u$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')
plt.show(block = False)

fig2 = plt.figure()
fig2.suptitle("Frequency Space (Magnitude)")
ax2 = fig2.add_subplot(projection = '3d')
img2 = ax2.scatter(XYZ[0], XYZ[1], XYZ[2], c = np.log10(U_mag + 1), cmap = 'Blues', alpha = 0.7)
fig2.colorbar(img2, location = 'left', label = r'$\log_{10} \left[ ||\hat{u}|| + 1 \right]$')
ax2.set_xlabel(r'$\hat{x}$')
ax2.set_ylabel(r'$\hat{y}$')
ax2.set_zlabel(r'$\hat{z}$')
plt.show(block = False)

fig3 = plt.figure()
fig3.suptitle("Frequency Space (Phase)")
ax3 = fig3.add_subplot(projection = '3d')
img3 = ax3.scatter(XYZ[0], XYZ[1], XYZ[2], c = U_phase, cmap = 'Blues', alpha = 0.1)
fig3.colorbar(img3, location = 'left', label = r'$\arctan2 \left[ \frac{\Im(\hat{u})}{\Re(\hat{u})} \right]$')
ax3.set_xlabel(r'$\hat{x}$')
ax3.set_ylabel(r'$\hat{y}$')
ax3.set_zlabel(r'$\hat{z}$')
plt.show()