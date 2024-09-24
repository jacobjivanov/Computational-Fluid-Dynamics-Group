#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>

// gcc "FFTW Test 3.c" -o "FFTW Test 3" -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3

int main(int argc, char *argv[])
{
   double time[] = {0.0000, 0.1282, 0.2565, 0.3847, 0.5129, 0.6411, 0.7694, 0.8976, 1.0258, 1.1541, 1.2823, 1.4105, 1.5387, 1.6670, 1.7952, 1.9234, 2.0517, 2.1799, 2.3081, 2.4363, 2.5646, 2.6928, 2.8210, 2.9493, 3.0775, 3.2057, 3.3339, 3.4622, 3.5904, 3.7186, 3.8468, 3.9751, 4.1033, 4.2315, 4.3598, 4.4880, 4.6162, 4.7444, 4.8727, 5.0009, 5.1291, 5.2574, 5.3856, 5.5138, 5.6420, 5.7703, 5.8985, 6.0267, 6.1550, 6.2832, }; // s
   double in[] = {0.0000, 0.1206, 0.2283, 0.3255, 0.4140, 0.4954, 0.5706, 0.6406, 0.7060, 0.7674, 0.8252, 0.8798, 0.9317, 0.9809, 1.0279, 1.0728, 1.1157, 1.1568, 1.1964, 1.2344, 1.2710, 1.3064, 1.3405, 1.3735, 2.4055, 2.4364, 2.4665, 2.4956, 2.5240, 2.5515, 2.5783, 2.6044, 2.6299, 2.6547, 2.6789, 2.7026, 2.7257, 2.7482, 2.7703, 2.7919, 2.8131, 2.8338, 2.8540, 2.8739, 2.8934, 2.9125, 2.9313, 2.9497, 2.9678, 2.9856, };

   int N = sizeof(in) / sizeof(in[0]);
   // printf("%d\n", N);

   fftw_complex *out;
   fftw_plan p;

   out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

   p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

   fftw_execute(p);
   int i;
   for (i = 0; i < N / 2 + 1; i++) {
      // printf("%lf\n", cabs(out[i]));
      printf("out[%d] = {%lf, %lfi}\n", i, creal(out[i]), cimag(out[i]));
   }

   fftw_destroy_plan(p);
   fftw_free(out);

   return 1;
}