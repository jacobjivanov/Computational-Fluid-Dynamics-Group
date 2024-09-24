// fourier_interpolate_v4.c
// This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. 

# include <stdio.h>
# include <complex.h>
// # include <fftw3.h>
# include <stdlib.h>
# include <math.h>

double coords[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

double inter_1D(float64 coords[], values[], pos) {
   int Ni = sizeof(coords) / sizeof(coords[0]);
   printf("%d\n", Ni);
}