/*
It is always a good practice to assign a NULL value to a pointer variable in case you do not have an exact address to be assigned. This is done at the time of variable declaration. A pointer that is assigned NULL is called a null pointer.

The NULL pointer is a constant with a value of zero defined in several standard libraries. Consider the following program:
*/

#include <stdio.h>

int main () {

   int  *ptr = NULL;

   printf("The value of ptr is : %x\n", ptr);
 
   return 0;
}

/*
gcc "tutorial 1-3.c" -o "tutorial 1-3.app"
./"tutorial 1-3.app"
*/

/*
The value of ptr is : 0
*/