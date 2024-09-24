# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

import numpy as np
import numba

def wavenumbers(N, dom):
    return np.array(list(range(0, N//2 + 1)) + list(range(- N//2 + 1, 0))) * 2 * np.pi / (dom[1] - dom[0])

@numba.njit
def inter_1D(kx, values1D_fft, pos):
    Ni, = values1D_fft.shape
    value_inter = 0

    for f in range(0, Ni):
        value_inter += values1D_fft[f] * np.exp(1j * kx[f] * pos)

    value_inter /= Ni
    return value_inter

def inter_2D(kx, ky, values2D_fft, pos):
    Ni, Nj = values2D_fft.shape

    inter_y_values = np.zeros(Ni, dtype = 'complex')
    for i in range(0, Ni):
        inter_y_values[i] = inter_1D(ky, values2D_fft[i], pos[1])
        # print(inter_y_values[i])
    
    inter_yx_value = inter_1D(kx, inter_y_values, pos[0])
    return inter_yx_value


def inter_3D(kx, ky, kz, values3D_fft, pos):
    Ni, Nj, Nk = values3D_fft.shape
    
    inter_zy_values = np.zeros(Ni, dtype = 'complex')

    for i in range(0, Ni):
        inter_y_values = np.zeros(Ni, dtype = 'complex')
        for j in range(0, Nj):
            inter_y_values[j] = inter_1D(kz, values3D_fft[i][j], pos[2])
        inter_zy_values[i] = inter_1D(ky, inter_y_values, pos[1])
    inter_zyx_value = inter_1D(kx, inter_zy_values, pos[0])
    return inter_zyx_value