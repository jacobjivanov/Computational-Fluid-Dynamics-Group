import numpy as np
import ffti_v10 as fi9

def f(x, y, z):
    # return np.exp(np.sin(x) - np.cos(z * 3 * y))
    return np.exp(np.sin(x) + np.sin(y) + np.sin(z))

def l2error(Ni):
    x_dom, Ni = [0, 2 * np.pi], Ni
    y_dom, Nj = [0, 2 * np.pi], Ni
    z_dom, Nk = [0, 2 * np.pi], Ni

    x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
    y = np.linspace(y_dom[0], y_dom[1], Nj, endpoint = False)
    z = np.linspace(z_dom[0], z_dom[1], Nk, endpoint = False)

    kx = fi9.wavenumbers(Ni, x_dom)
    ky = fi9.wavenumbers(Nj, y_dom)
    kz = fi9.wavenumbers(Nk, z_dom)

    rho = np.empty(shape = (Ni, Nj, Nk))
    for i in range(0, Ni):
        for j in range(0, Nj):
            for k in range(0, Nk):
                rho[i, j, k] = f(x[i], y[j], z[k])
    rho_fft = np.fft.fftn(rho)


    Ni_inter = 100
    Nj_inter = 100
    Nk_inter = 100

    x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter, endpoint = False)
    y_inter = np.linspace(x_dom[0], x_dom[1], Nj_inter, endpoint = False)
    z_inter = np.linspace(x_dom[0], x_dom[1], Nk_inter, endpoint = False)

    s = 0
    for ii in range(0, Ni_inter):
        for ji in range(0, Nj_inter):
            for ki in range(0, Nk_inter):
                a = np.real(fi9.inter_3D(kx, ky, kz, rho_fft, [x_inter[ii], y_inter[ji], z_inter[ki]])) - f(x_inter[ii], y_inter[ji], z_inter[ki])
                s += a ** 2
            print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (ii + ((ji + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
    print("Interpolating. \tProgress: 100.000% Complete")

    s *= (x_inter[1] * y_inter[1] * z_inter[1])/(Ni_inter * Nj_inter * Nk_inter)
    s = np.sqrt(s)

    print(Ni, s)

for n in range(4, 51):
    l2error(n)