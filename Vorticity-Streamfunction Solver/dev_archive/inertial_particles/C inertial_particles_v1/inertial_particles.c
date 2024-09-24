# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <complex.h>

const double pi = 3.14159265358979323846;

int main (int argc, char *argv[]) {
    int M = atoi(argv[1]); // computational grid dimensions
    double T = atof(argv[2]); // ending time
    int P = atoi(argv[3]); // particle count

    double nu = atof(argv[4]); // viscosity
    double tau = atof(argv[5]); // inertial time

    return 0;
}