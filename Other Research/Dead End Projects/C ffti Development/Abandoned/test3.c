# include <stdio.h>
# include <stdlib.h>

double* linspace();

int main() {
   const int Ni = 101, Nj = 101, Nk = 101;

   const double x_max = 10, y_max = 10, z_max = 10;

   double *x[Ni] = &linspace(1., x_max, Ni);
   double *y[Nj] = &linspace(1.3, y_max, Nj);
   double *z[Nk] = &linspace(1.5, z_max, Nk);

   printf("%f\n", y[1]);
   
   return 0;
}

double *linspace(double start, double stop, int n) {
   double *array = malloc(n * sizeof *array);
   for (int i = 0; i < n; i++) {
      array[i] = start + i * (stop - start) / (double)(n - 1);
   }
   return array;
}