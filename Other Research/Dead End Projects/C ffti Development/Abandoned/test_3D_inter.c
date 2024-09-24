#include "trifourier_v0.c"
#include <math.h>

double func(double x, double y, double z) {
   return pow(M_E, sin(x) + sin(y) + sin(z));
}

int main() {
   const int Ni = 32, Nj = 32, Nk = 32;
   const double x_max = 2 * M_PI, y_max = 2 * M_PI, z_max = 2 * M_PI;

   double *x = linspace(0, x_max, Ni);
   double *y = linspace(0, y_max, Nj);
   double *z = linspace(0, z_max, Nk);

   double (*rho)[Nj][Nk] = malloc(Ni * Nj * Nk * sizeof(double));

   for (int i = 0; i < Ni; i++) {
      for (int j = 0; j < Nj; j++) {
         for (int k = 0; k < Nk; k++) {
            // rho[i][j][k] = pow(M_E, sin(x[i]) + sin(y[j]) + sin(z[k]));
            rho[i][j][k] = func(x[i], y[j], z[k]);
            // printf("rho[%.5f][%.5f][%.5f] = %.16f\n", x[i], y[j], z[k], rho[i][j][k]);
         }
      }
   }
   
   const int Ni_inter = 4 * Ni, Nj_inter = 4 * Nj, Nk_inter = 4 * Nk;
   double *x_inter = linspace(0, x_max, Ni_inter);
   double *y_inter = linspace(0, y_max, Nj_inter);
   double *z_inter = linspace(0, z_max, Nk_inter);

   double a = inter_3D(x, y, z, Ni, Nj, Nk, rho, 1.57, 1.99, 2.52);
   double b = func(1.57, 1.99, 2.52);
   printf("rho_int(%.5f, %.5f, %.5f) = %.16f\n", 1.57, 1.99, 2.52, a);
   printf("rho_ana(%.5f, %.5f, %.5f) = %.16f\n", 1.57, 1.99, 2.52, b);

/*
   double rho_inter[Ni_inter][Nj_inter][Nk_inter];

   for (int i = 0; i < Ni_inter; i++) {
      for (int j = 0; j < Nj_inter; j++) {
         for (int k = 0; k < Nk_inter; k++) {
            rho_inter[i][j][k] = inter_3D(x, y, z, Ni, Nj, Nk, rho, x_inter[i], y_inter[j], z_inter[k]);
            printf("rho_inter[%d][%d][%d] = %.16f\n", i, j, k, rho_inter[i][j][k]);
         }
      }
   }

   double rho_inter_error[Ni_inter][Nj_inter][Nk_inter];

   for (int i = 0; i < Ni_inter; i++) {
      for (int j = 0; j < Nj_inter; j++) {
         for (int k = 0; k < Nk_inter; k++) {
            rho_inter_error[i][j][k] = rho_inter[i][j][k] - func(x_inter[i], y_inter[j], z_inter[k]);
            printf("rho_inter_error[%d][%d][%d] = %.16f\n", i, j, k, rho_inter_error[i][j][k]);
         }
      }
   }
   */

}