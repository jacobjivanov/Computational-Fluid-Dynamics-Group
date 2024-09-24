#include <stdio.h>

int main () {

   int  var = 20;   /* actual variable declaration */
   int  *ip;        /* pointer variable declaration */

   ip = &var;  /* store address of var in pointer variable*/

   printf("Address of var variable: %x\n", &var);

   /* address stored in pointer variable */
   printf("Address stored in ip variable: %x\n", ip);

   /* access the value using the pointer */
   printf("Value of *ip variable: %d\n", *ip);

   return 0;
}

/*
gcc  "tutorial 1-2.c" -o "tutorial 1-2.app"
./"tutorial 1-2.app"
*/

/*
Address of var variable: 6b62f0b8
Address stored in ip variable: 6b62f0b8
Value of *ip variable: 20
*/