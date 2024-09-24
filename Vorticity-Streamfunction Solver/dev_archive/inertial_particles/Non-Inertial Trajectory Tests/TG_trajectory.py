import numpy as np
import matplotlib.pyplot as plt

# simulation parameters
M, N = 128, 128 # computational grid size
nu = 5e-3 # kinematic viscosity
beta = 1 # periods within the [0, 2π) domain
t_end = 200
dt = 0.5

x, y = np.linspace(0, 2*np.pi, M, endpoint = False), np.linspace(0, 2*np.pi, N, endpoint = False)
x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
dx, dy = x[1], y[1]
eo = M // 32

def U_TG(t):
    return np.fft.fft2(np.exp(-2 * nu * t) * np.cos(beta * x_grid) * np.sin(beta * y_grid))

def V_TG(t):
    return np.fft.fft2(-np.exp(-2 * nu * t) * np.sin(beta * x_grid) * np.cos(beta * y_grid))

def u_TG(t, x, y):
    return np.exp(-2 * nu * t) * np.cos(beta * x) * np.sin(beta * y)

def v_TG(t, x, y):
    return -np.exp(-2 * nu * t) * np.sin(beta * x) * np.cos(beta * y)

def wavearrays(M, N):
    k_p = np.tile(
        np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0))), reps = (N, 1)).T

    k_q = np.tile(
        np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))), reps = (M, 1))

    return k_p, k_q

kp, kq = wavearrays(M, N)

def inter_2D(U, x_pos, y_pos):
    M, N = U.shape

    u_inter = 0
    for p in range(0, M):
        u_yinter = 0
        for q in range(0, N):
            u_yinter += U[p, q] * np.exp(1j * kq[p, q] * y_pos)

        u_yinter /= N
        u_inter += u_yinter * np.exp(1j * kp[p, 0] * x_pos)

    u_inter /= M
    return u_inter

x_part = np.random.rand()
y_part = np.random.rand()

# the following implements first-order Euler Integration for particle position. It was shown that the analytical value for u, v, and the spectrally interpolated values based on the computational mesh agreed to within 1e-8 even after t = 200. The interpolation for v needs to use the last x position, not the next one, so the lines can't naively just follow each other

x_part_dir, y_part_dir = x_part, y_part

t = 0
while t < t_end:
    u, v = np.real(np.fft.ifft2(U_TG(t))), np.real(np.fft.ifft2(V_TG(t)))
    # uv_mag = np.sqrt(u * u + v * v)

    # plt.pcolormesh(x_grid, y_grid, uv_mag, cmap = 'coolwarm')
    # plt.quiver(x_grid[::eo, ::eo], y_grid[::eo, ::eo], u[::eo, ::eo], v[::eo, ::eo])
    # plt.scatter(x_part, y_part)
    plt.show()

    u_part_dir = u_TG(t, x_part_dir, y_part_dir)
    v_part_dir = v_TG(t, x_part_dir, y_part_dir)

    x_part_dir = (x_part_dir + dt * u_part_dir) % (2 * np.pi)
    y_part_dir = (y_part_dir + dt * v_part_dir) % (2 * np.pi)

    u_part = np.real(inter_2D(U_TG(t), x_part, y_part))
    v_part = np.real(inter_2D(V_TG(t), x_part, y_part))
    
    x_part = (x_part + dt * u_part) % (2 * np.pi)
    y_part = (y_part + dt * v_part) % (2 * np.pi)

    print(x_part - x_part_dir, y_part - y_part_dir)
    t += dt