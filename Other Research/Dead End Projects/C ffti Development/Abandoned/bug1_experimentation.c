// This file is an experimental file meant to track down the bug in `trifourier.c` that causes seemingly random behavior when Ni = Nj = Nk = 16, or where Ni != Nj != Nk

// gcc bug1_experimentation.c -o bug1_experimentation.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./bug1_experimentation.out

#include "trifourier_v0.c"
#include <math.h>

int main() {
   int Ni = 16;
   double *x = linspace(0, 2 * M_PI, Ni);

   double *y = malloc(16 * sizeof(double));


   for (int i = 0; i < Ni; i++) {
      y[i] = pow(M_E, sin(x[i]));
      // printf("x[%d] = %.5f, \ty[%d] = %.5f\n", i, x[i], i, y[i]);
   }

   double x_p = 1.3;
   double a = inter_1D(x, y, Ni, x_p);
   printf("%f\n", a);
   printf("%f\n", pow(M_E, sin(1.3)));
}

// 04/09 7:12 PM, Based on testing with the Python `bug1_experimentation.py`, I have noticed that line 24 outputs `3.669297` for C, `2.6210059262866956` for py. Nevermind. Typo.