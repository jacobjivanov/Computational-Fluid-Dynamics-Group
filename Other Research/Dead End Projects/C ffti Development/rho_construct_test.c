#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double *linspace(double start, double stop, int n) {
   double *array = malloc(n * sizeof *array);
   for (int i = 0; i < n; i++) {
      array[i] = start + i * (stop - start) / (n - 1);
   }
   return array;
}

int main() {
   const int Ni = 21, Nj = 21, Nk = 21;
   const double x_max = 2 * M_PI, y_max = 2 * M_PI, z_max = 2 * M_PI;

   double *x = linspace(0, x_max, Ni);
   double *y = linspace(0, y_max, Nj);
   double *z = linspace(0, z_max, Nk);

   // for (int i = 0; i < Ni; i++) {
   //    printf("%f\n", x[i]);
   // }

   double rho[Ni][Nj][Nk];

   for (int i = 0; i < Ni; i++) {
      for (int j = 0; j < Nj; j++) {
         for (int k = 0; k < Nk; k++) {
            rho[i][j][k] = pow(M_E, sin(x[i]) + sin(y[j]) + sin(z[k]));
            printf("rho[%d][%d][%d] = %f\n", i, j, k, rho[i][j][k]);
         }
      }
   }
}