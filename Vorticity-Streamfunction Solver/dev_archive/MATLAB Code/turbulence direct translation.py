# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
import ffti_v10 as fi
import netCDF4
import matplotlib.pyplot as plt

x_dom, y_dom = [0, 2 * np.pi], [0, 2 * np.pi]
M, N = 256, 256
dx, dy = (x_dom[1] - x_dom[0])/M, (y_dom[1] - y_dom[0])/N

x = np.linspace(x_dom[0], x_dom[1], M, endpoint = False)
y = np.linspace(y_dom[0], y_dom[1], N, endpoint = False)
kx, ky = fi.wavenumbers(M, x_dom), fi.wavenumbers(N, y_dom)

time = 0

nu = 5e-4
Sc = 0.7
beta = 0
ar = 0.02
b = 1
CFLmax = 0.8
tend = 0.05

index_kmax = int(np.ceil(M / 3))
kmax = kx[index_kmax]

fil = np.ones(shape = (M, N))
fil[index_kmax:2 * index_kmax + 4, index_kmax:2 * index_kmax + 4] = 0

u = np.zeros(shape = (M, N), dtype = 'complex')
v = np.zeros(shape = (M, N), dtype = 'complex')
omega = np.zeros(shape = (M, N), dtype = 'complex')
psi = np.zeros(shape = (M, N), dtype = 'complex')
ddx = np.zeros(shape = (M, N), dtype = 'complex')
ddy = np.zeros(shape = (M, N), dtype = 'complex')
idel2 = np.zeros(shape = (M, N), dtype = 'complex')
kk = np.zeros(shape = (M, N), dtype = 'complex')
k2 = np.zeros(shape = (M, N), dtype = 'complex')

for j in range(0, N):
    ddx[:, j] = 1j * kx
for i in range(0, M):
    ddy[i, :] = 1j * ky

for i in range(0, M):
    for j in range(0, N):
        idel2[i, j] = -kx[i]**2 - ky[j]**2
idel2 = np.reciprocal(idel2)
idel2[0, 0] = 0

# DEBUGGING
print(idel2)

for i in range(0, M):
    for j in range(0, N):
        kk[i, j] = kx[i]**2 + ky[j]**2
        k2[i, j] = kx[i]**2 + ky[j]**2

        if 6**2 <= kk[i, j] <= 7**2: # forcing
            kk[i, j] *= -1
        
        if kk[i, j] <= 2**2: # large-scale dissipation
            kk[i, j] *= 8

for i in range(0, M):
    for j in range(0, N):
        u[i, j] =  np.cos(2 * x[i]) * np.sin(2 * y[j]) + ar * np.random.random()
        u[i, j] =  - np.sin(2 * x[i]) * np.cos(2 * y[j]) + ar * np.random.random()

# DEBUGGING
uv_mag = (u**2 + v**2) ** 0.5
plt.contourf(x, y, uv_mag)
plt.show()

uhat = np.fft.fft2(u)
vhat = np.fft.fft2(v)
omegahat = ddx * vhat - ddy * uhat # make vorticity

phi = np.random.rand(M, N)
phihat = np.fft.fft2(phi)

"""
ncid = netCDF4.Dataset('turb2d.nc', mode = 'w')
dimid_x = ncid.createDimension('x', M)
dimid_y = ncid.createDimension('y', N)
dimid_time = ncid.createDimension('time', None)

varid_x = ncid.createVariable('x', np.float32, (dimid_x,))
varid_y = ncid.createVariable('y', np.float32, (dimid_y,))
varid_u = ncid.createVariable('u', np.float32, (dimid_x, dimid_y, dimid_time))
varid_v = ncid.createVariable('v', np.float32, (dimid_x, dimid_y, dimid_time))
varid_omega = ncid.createVariable('vorticity', np.float32, (dimid_x, dimid_y, dimid_time))
varid_phi = ncid.createVariable('scalar', np.float32, (dimid_x, dimid_y, dimid_time))
varid_dissipation = ncid.createVariable('dissipation', np.float32, (dimid_x, dimid_y, dimid_time))
varid_time = ncid.createVariable('time', np.float32, (dimid_time,))


varid_time[0] = time
varid_x[0:M] = x
varid_y[0:N] = y
varid_phi[0:M, 0:N, 0] = phi
varid_u[0:M, 0:N, 0] = u
varid_v[0:M, 0:N, 0] = v
varid_omega[0:M, 0:N, 0] = omega
varid_dissipation[0:M, 0:N, 0] = 0 * omega
ncid.close()
"""

dt = 0.5 * np.min([dx, dy])
nstep = 1

while time < tend:
    psihat = - idel2 * omegahat
    uhat = ddy * psihat
    vhat = - ddx * psihat

    u = np.real(np.fft.ifft2(uhat))
    v = np.real(np.fft.ifft2(vhat))
    
    omegadx = np.real(np.fft.ifft2(ddx * omegahat))
    omegady = np.real(np.fft.ifft2(ddy * omegahat))

    facto = np.exp(- nu * 8/15 * dt * kk)
    factp = np.exp(- nu/Sc * 8/15 * dt * k2)

    r0o = - np.fft.fft2(u * omegadx + v * omegady) + beta * vhat
    r0p = - np.fft.fft2(u * np.real(np.fft.ifft2(ddx * phihat)) + v * np.real(np.fft.ifft2(ddy * phihat))) + b * vhat

    omegahat = facto * (omegahat + dt * 8/15 * r0o)
    ohihat = factp * (phihat + dt * 8/15 * r0p)

    # Substep 2
    psihat = - idel2 * omegahat
    uhat = ddy * psihat
    vhat = -ddx * psihat

    u = np.real(np.fft.ifft2(uhat))
    v = np.real(np.fft.ifft2(vhat))

    omegadx = np.real(np.fft.ifft2(ddx * omegahat))
    omegady = np.real(np.fft.ifft2(ddy * omegahat))

    r1o = - np.fft.fft2(u * omegadx + v * omegady) + beta * vhat
    r1p = - np.fft.fft2(u * np.real(np.fft.ifft2(ddx * phihat) + v * np.real(np.fft.ifft2(ddy * phihat)))) + b * vhat

    omegahat += dt * (-17/60 * facto * r0o + 5/12 * r1o)
    phihat += dt * (-17/60 * factp * r0p + 5/12 * r1p)
    facto = np.exp(-nu * (-17/60 + 5/12) * dt * kk)
    factp = np.exp(-nu/Sc * (-17/60 + 5/12) * dt * k2)
    omegahat *= facto
    phihat *= factp

    # Substep 3
    psihat = - idel2 * omegahat
    uhat = ddy * psihat
    vhat = -ddx * psihat

    u = np.real(np.fft.ifft2(uhat))
    v = np.real(np.fft.ifft2(vhat))

    omegadx = np.real(np.fft.ifft2(ddx * omegahat))
    omegady = np.real(np.fft.ifft2(ddy * omegahat))

    r2o = -np.fft.fft2(u * omegadx + v * omegady) + beta * vhat
    r2p = -np.fft.fft2(u * np.real(np.fft.ifft2(ddx * phihat) + v * np.real(np.fft.ifft2(ddy * phihat)))) + b * vhat
    omegahat += dt * (-5/12 * facto * r1o * 3/4 * r2o)
    phihat += dt * (-5/12 * factp * r1p * 3/4 * r2p)
    facto = np.exp(-nu * (-5/12 + 3/4) * dt * kk)
    factp = np.exp(-nu/Sc * (-5/12 + 3/4) * dt * kk)
    omegahat *= facto
    phihat *= factp

    phihat *= fil
    omegahat *= fil

    CFL = np.amax(u)/dx * dt * np.amax(v)/dy * dt

    time += dt
    nstep += 1

    if nstep % 1 == 0:
        uv_mag = (u**2 + v**2) ** 0.5
        # print(uv_mag)
        plt.contourf(x, y, uv_mag)
        plt.show()
    print(time)