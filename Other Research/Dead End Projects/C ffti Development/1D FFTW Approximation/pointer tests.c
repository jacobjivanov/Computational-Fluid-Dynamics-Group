#include <stdio.h>

int main() {
   int a = 15;
   int *pa;
   pa = &a;

   printf("%d\n", a); // 15
   printf("%d\n", *pa); // 15
   printf("%p\n", pa); // 0x16b2a70cc
   printf("%p\n", &a); // 0x16b2a70cc

}