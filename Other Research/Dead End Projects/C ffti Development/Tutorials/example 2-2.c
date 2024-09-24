# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>

double avg();

int main() {
   const int N = 1000000;
   double array[N] = {0};
   
   srand (time(NULL));
   for (int i = 0; i < N; i ++) {
      array[i] = (float)rand() / (float)RAND_MAX;
   }
   
   double a = avg(array, N);
   printf("%0.16lf\n", a);
}

double avg(double *array, int n) {
   double sum = 0;
   for (int i = 0; i < n; i ++) {
      sum += array[i];
   }
   return sum / (double)n;
}

/*
gcc "example 2-2.c" -o "example 2-2.app"
./"example 2-2.app"
*/

/*
0.4998271209545657
*/