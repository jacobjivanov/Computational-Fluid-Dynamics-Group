# This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut.

# the goal of this program is to evolve the vorticity transport equation initialized as perturbed Taylor-Green Vortex Flow. Due to the computational expensiveness inertial particles, MPI will be utilized to parallelize the transport of individual particles. Additionally, numpy.fft.rfft()-type operations will be utilized over numpy.fft.fft()-type operations in order to speed it up. Checkpoint Restart will also be implemented.

import numpy as np
from numpy import real, imag
from numpy.fft import rfft2, irfft2

import numba
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def init_grid(M):
    
def init_wavearrays(M, nu):

def init_Omega(M, beta):

def init_P(N):

def calc_inter2D(U, x, y):

def calc_Xi(t):

def update_Omega():

def update_P()

def calc_TKE(Omega):

def save_animation():

def restart():

if __name__ == "__main__":
    if rank == 0: # main node

    if rank != 0: # satellite node