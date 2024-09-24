// This program was written in its entirety by Jacob Ivanov, Undergraduate Research Assistant under Dr. Georgios Matheou for the Computational Fluid Dynamics Group at the University of Connecticut. 

// This program is my attempt to replicate functionality of the `ffti_v6.py` (exclusively zyx order trifourier interpolation) library

// gcc ffti_v5.c -o ffti_v5.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./ffti_v5.out

#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double *linspace(double start, double stop, int n);
double inter_1D(double *coords, double *values, int Ni, double pos);
double inter_3D(double *x_coords, double *y_coords, double *z_coords, int Ni, int Nj, int Nk, double (*values3D)[Nj][Nk], double x_pos, double y_pos, double z_pos);

double *linspace(double start, double stop, int n) {
   double *array = malloc(n * sizeof *array);
   for (int i = 0; i < n; i++) {
      array[i] = start + i * (stop - start) / (n - 1);
   }
   return array;
}

double inter_1D(double *coords, double *values, int Ni, double pos) {
   Ni -= 1;
   const double c_max = coords[Ni];
   // printf("LINE 52 CMAX = %f\n", c_max);
   // printf("LINE 53 COORDS[1] = %f\n", coords[1]);
   fftw_complex *values_fft;
   fftw_plan p;

   values_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (Ni / 2 + 1));

   // Makes no difference
   // double *values_sliced = malloc((Ni) * sizeof(double));
   // for (int i = 0; i < Ni; i++) {
   //    values_sliced[i] = values[i];
   // }
   // p = fftw_plan_dft_r2c_1d(Ni, values_sliced, values_fft, FFTW_ESTIMATE);

   p = fftw_plan_dft_r2c_1d(Ni, values, values_fft, FFTW_ESTIMATE);
   fftw_execute(p);

   double value_inter = creal(values_fft[0]);
   for (int f = 1; f < (Ni + 1) / 2 + 1; f++) {
      double w = 2 * f * M_PI / c_max;
      value_inter += 2 * creal(values_fft[f]) * cos(w * pos);
      value_inter -= 2 * cimag(values_fft[f]) * sin(w * pos);
   }
   value_inter /= Ni;

   return value_inter;
}

double inter_3D(double *x_coords, double *y_coords, double *z_coords, int Ni, int Nj, int Nk, double (*values3D)[Nj][Nk], double x_pos, double y_pos, double z_pos) {
   // order zyx

   double *d1_coords = malloc(Nk * sizeof(double));
   double *d2_coords = malloc(Nj * sizeof(double));
   double *d3_coords = malloc(Ni * sizeof(double));

   memcpy(d1_coords, z_coords, Nk * sizeof(double));
   memcpy(d2_coords, y_coords, Nj * sizeof(double));
   memcpy(d3_coords, x_coords, Ni * sizeof(double));

   double pos_d1 = z_pos;
   double pos_d2 = y_pos;
   double pos_d3 = x_pos;

   int Nd1 = Ni;
   // printf("%d\n", Nd1);
   int Nd2 = Nj;
   // printf("%d\n", Nd1);
   int Nd3 = Nk;
   // printf("%d\n", Nd1);


   double inter_d1d2_values[Nd3];
   for (int d3 = 0; d3 < Nd3; d3++) {
      double inter_d2_values[Nd2];
      for (int d2 = 0; d2 < Nd2; d2++) {
         // inter_d2_values[d2] = inter_1D(d1_coords, values3D[d3][d2], Nd1, pos_d1);
         inter_d2_values[d2] = inter_1D(d1_coords, values3D[d3][d2], Nd1, pos_d1);
         // printf("LINE 102 %f\n", values3D[d3][1][2]);
         // printf("LINE 103 %f\n", inter_d2_values[d2]);

      }
      inter_d1d2_values[d3] = inter_1D(d2_coords, inter_d2_values, Nd2, pos_d2);
      // printf("LINE 101 %f\n", inter_d1d2_values[d3]);
   }
   double inter_d1d2d3_value = inter_1D(d3_coords, inter_d1d2_values, Nd3, pos_d3);
   free(d1_coords); free(d2_coords); free(d3_coords);
   return inter_d1d2d3_value;
}