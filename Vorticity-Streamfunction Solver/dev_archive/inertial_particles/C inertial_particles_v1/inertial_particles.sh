#!/bin/sh

exec rm inertial_particles.out
exec gcc inertial_particles.c -o inertial_particles.out
exec ./inertial_particles.out 256 200 50 9e-4 0.5