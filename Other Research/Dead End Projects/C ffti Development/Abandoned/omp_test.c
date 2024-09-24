#include <stdio.h>
#include <omp.h>

// gcc -I/opt/homebrew/Cellar/libomp/15.0.7/include -L/opt/homebrew/Cellar/libomp/15.0.7/lib omp_test.c -o omp_test.out -lomp

int main() {
   #pragma omp parallel 
   {
      printf(" Hello ");
      printf(" World \n");
   }
}