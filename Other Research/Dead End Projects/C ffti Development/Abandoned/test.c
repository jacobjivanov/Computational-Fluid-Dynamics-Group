# include <stdio.h>
void disp4(int num[2][2]) {
   printf("Displaying:\n");
   for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
         printf("%d\n", num[i][j]);
      }
   }
}

int main() {
   int num[2][2];
   printf("Enter 4 numbers:\n");
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
         scanf("%d", &num[i][j]);
      }
   }

   // pass multi-dimensional array to a function  
   disp4(num);
   return 0;
}