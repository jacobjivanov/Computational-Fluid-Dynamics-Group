// This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. 

// This program is my attempt to replicate functionality of the `inter_1D()` function from `ffti_v5.py`

// gcc inter_1Dv0.c -o inter_1Dv0.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./inter_1Dv0.out

#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double inter_1D(double *coords, double *values, int Ni, double pos) {
   Ni -= 1;
   const double c_max = coords[Ni];
   // printf("%f\n", c_max);
   fftw_complex *values_fft;
   fftw_plan p;

   values_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (Ni / 2 + 1));

   p = fftw_plan_dft_r2c_1d(Ni, values, values_fft, FFTW_ESTIMATE);
   fftw_execute(p);

   double value_inter = creal(values_fft[0]);
   for (int f = 1; f < (Ni + 1) / 2 + 1; f++) {
      double w = f * 2 * M_PI / c_max;
      value_inter += 2 * creal(values_fft[f]) * cos(w * pos);
      value_inter -= 2 * cimag(values_fft[f]) * sin(w * pos);
   }
   value_inter /= Ni;

   return value_inter;
}

int main() {
   double time[] = {0.0,0.5714285714285714,1.1428571428571428,1.7142857142857142,2.2857142857142856,2.8571428571428568,3.4285714285714284,4.0,4.571428571428571,5.142857142857142,5.7142857142857135,6.285714285714286,6.857142857142857,7.428571428571428,8.0,};
   
   double signal[] = {0.0,0.5408342133588315,0.9098229129411239,0.9897230488598214,0.7551470262316581,0.2806293995143573,-0.2830558540822556,-0.7568024953079283,-0.9900815210958355,-0.9087704868046733,-0.538705288386157,0.002528975838921635,0.5429596793024328,0.910869520096767,0.9893582466233818,};
   
   int Ni = sizeof(time) / sizeof(time[0]);
   double a = inter_1D(time, signal, Ni, 5.76112);
   printf("%.16f\n", a);
}