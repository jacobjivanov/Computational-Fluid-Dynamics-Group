#include <stdio.h>

int main () {

   int  var1;
   char var2[10];

   printf("Address of var1 variable: %x\n", &var1);
   printf("Address of var2 variable: %x\n", &var2);

   return 0;
}

/*
gcc "tutorial 1-1.c" -o "tutorial 1-1.app" 
./"tutorial 1-1.app"
*/

/*
Address of var1 variable: 6cf6f0a4
Address of var2 variable: 6cf6f0ae
*/