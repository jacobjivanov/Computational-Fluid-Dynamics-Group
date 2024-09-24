#include <complex.h>
#include <fftw3.h> // ignore red underline, VS Code is dumb
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TO COMPILE: 
// gcc "FFTW 1D Approximation.c" -o "FFTW 1D Approximation" -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3

// TO EXECUTE: 
// ./"FFTW 1D Approximation"

int main(int argc, char *argv[]) {
   // time and signal arrays should be equally spaced and coorespond to one another
   double time[] = {0.0,0.5714285714285714,1.1428571428571428,1.7142857142857142,2.2857142857142856,2.8571428571428568,3.4285714285714284,4.0,4.571428571428571,5.142857142857142,5.7142857142857135,6.285714285714286,6.857142857142857,7.428571428571428,8.0,};
   
   double signal[] = {0.0,0.5408342133588315,0.9098229129411239,0.9897230488598214,0.7551470262316581,0.2806293995143573,-0.2830558540822556,-0.7568024953079283,-0.9900815210958355,-0.9087704868046733,-0.538705288386157,0.002528975838921635,0.5429596793024328,0.910869520096767,0.9893582466233818,};

   const int N = sizeof(signal) / sizeof(signal[0]);
   const double t_max = time[N - 1];

   fftw_complex *signal_fft;
   fftw_plan p;

   signal_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));

   // TODO: REMOVE LAST INDEX PRIOR TO FFT
   p = fftw_plan_dft_r2c_1d(N, signal, signal_fft, FFTW_ESTIMATE);

   fftw_execute(p);

   double signal_approx[N] = {0};
   for (int t = 0; t < N; t++) {
      // FFTW3 does not return both conjugates, and the first element is the real "offset"
      signal_approx[t] += creal(signal_fft[0]) / N;
      for (int i = 1; i < N / 2 + 1; i++) {
         // `2 * M_PI / T_max` scales from default 2 pi length to any time series
         signal_approx[t] += (2 * creal(signal_fft[i]) * cos(i * 2 * M_PI * time[t] / t_max) - 2 * cimag(signal_fft[i]) * sin(i * 2 * M_PI * time[t] / t_max)) / N;
      }
   }

   for (int i = 0; i < N; i++) {
      printf("%lf\n", signal_approx[i]);
   }

   fftw_destroy_plan(p);
   fftw_free(signal_fft);

   return 1;
}