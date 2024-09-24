# include <stdio.h>
# include <stdlib.h>

double* linspace();

int main() {
   const int Ni = 101, Nj = 101, Nk = 101;
   const double x_max = 10, y_max = 10, z_max = 10;

   double* x = linspace(0, x_max, Ni);
   double* y = linspace(0, y_max, Nj);
   double* z = linspace(0, z_max, Nk);

   double rho[Ni][Nj][Nk];

   for (int i = 0; i < Ni; i++) {
      for (int j = 0; j < Nj; j++) {
         for (int k = 0; k < Nk; k++) {
            rho[i][j][k] = x[i] + y[j] + z[k];
         }
      }
   }

   printf("%f\n", y[1]);
   // for (int i = 0; i < Ni; i++) {
   //    for (int j = 0; j < Nj; j++) {
   //       for (int k = 0; k < Nk; k++) {
   //          printf("rho[%d][%d][%d] = %f\n", i, j, k, rho[i][j][k])
   //       }
   //    }
   // }
   
   return 0;
}

double* linspace(double start, double stop, int n) {
   double* array = malloc(n * sizeof *array);
   for (int i = 0; i < n; i++) {
      array[i] = start + i * (stop - start) / n;
   }
   return array;
}