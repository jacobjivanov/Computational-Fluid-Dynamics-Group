#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// gcc linspace_test.c -o linspace_test.out
// ./linspace_test.out

double *linspace(double, double, int);

int main() {    
   double *x = linspace(0, 5, 6);
   double *y = linspace(0, 1.256, 12);
   double *z = linspace(0, 100, 15);

   printf("%f\n", x[2]);
}

double *linspace(double start, double stop, int n) {
   double *array = malloc(n * sizeof (double));
   for (int i = 0; i < n; i++) {
      array[i] = start + i * (stop - start) / n;
   }
   return array;
}