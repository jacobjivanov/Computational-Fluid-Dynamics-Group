#!/bin/sh

exec gcc inertial_particles.c -o inertial_particles.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
exec ./inertial_particles.out > inertial_particles.csv
# echo python3 inertial_particles_animation.py