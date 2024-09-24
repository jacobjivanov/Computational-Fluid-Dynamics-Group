def ga(Ni):
    import ffti_v10 as fi10
    import li_v6 as li6
    import numpy as np

    def func(x, y, z):
        rho = np.exp(np.sin(x) + np.sin(y) + np.sin(z))
        return rho
   
    Nj = Ni
    Nk = Ni

    x_dom = [- np.pi, np.pi]
    y_dom = [- np.pi, np.pi]
    z_dom = [- np.pi, np.pi]

    x = np.linspace(x_dom[0], x_dom[1], Ni, endpoint = False)
    y = np.linspace(y_dom[0], y_dom[1], Nj, endpoint = False)
    z = np.linspace(z_dom[0], z_dom[1], Nk, endpoint = False)

    kx = fi10.wavenumbers(Ni, x_dom)
    ky = fi10.wavenumbers(Nj, y_dom)
    kz = fi10.wavenumbers(Nk, z_dom)

    rho = np.zeros(shape = (Ni, Nj, Nk))
    for i in range(Ni):
        for j in range(Nj):
            for k in range(Nk):
                rho[i, j, k] = func(x[i], y[j], z[k])
    rho_fft = np.fft.fftn(rho)

    np.save("Convergence Test Outputs/rho[Ni = {}].npy".format(Ni), rho)

    Ni_inter = 4 * Ni
    Nj_inter = 4 * Nj
    Nk_inter = 4 * Nk

    x_inter = np.linspace(x_dom[0], x_dom[1], Ni_inter)
    y_inter = np.linspace(y_dom[0], y_dom[1], Nj_inter)
    z_inter = np.linspace(z_dom[0], z_dom[1], Nk_inter)

    # rho_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter), dtype = 'complex')
    rho_fi10 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter), dtype = 'complex')

    # error_li6 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter), dtype = 'complex')
    error_fi10 = np.zeros(shape = (Ni_inter, Nj_inter, Nk_inter), dtype = 'complex')

    for i in range(Ni_inter):
        for j in range(Nj_inter):
            for k in range(Nk_inter):
                # rho_li6[i, j, k] = li6.inter_3D(x, y, z, rho, pos = [x_inter[i], y_inter[j], z_inter[k]])
                rho_fi10[i, j, k] = fi10.inter_3D(kx, ky, kz, rho_fft, pos = [x_inter[i], y_inter[j], z_inter[k]])

                # error_li6[i, j, k] = rho_li6[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])
                error_fi10[i, j, k] = rho_fi10[i, j, k] - func(x_inter[i], y_inter[j], z_inter[k])

            print("Interpolating. \tProgress: {0:07.3f}% Complete".format(100 * (i + ((j + 1 ) / Nj_inter) ) / Ni_inter), end = '\r')
    print("Interpolating. \tProgress: 100.000% Complete")
   
    # np.save("Convergence Test Outputs/rho_li6[Ni = {}].npy".format(Ni), rho_li6)
    np.save("Convergence Test Outputs/rho_fi10[Ni = {}].npy".format(Ni), rho_fi10)
    # np.save("Convergence Test Outputs/error_li6[Ni = {}].npy".format(Ni), error_li6)
    np.save("Convergence Test Outputs/error_fi10[Ni = {}].npy".format(Ni), error_fi10)

ga(15)