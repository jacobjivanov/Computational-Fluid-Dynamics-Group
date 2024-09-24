# include <stdio.h>
# include <stdlib.h>
# include <math.h>

const double pi = 3.14159265358979323846;

double* kx1D_func(int M, double x_max) {
    double* kx = (double*)malloc(sizeof(double) * M);
    
    for (int p = 0; p <= M/2; p++) {
        kx[p] = 2*pi*p/x_max;
    }

    for (int p = M/2 + 1; p <= M; p++) {
        kx[p] = 2*pi*(p - M)/x_max;
    }

    return kx;
}

double* kx2D_func(int M, int N, double x_max) {
    double* kx = (double*)malloc(sizeof(double) * M * N);

    for (int p = 0; p < M/2 + 1; p++) {
        for (int q = 0; q < N; q++) {
            kx[M*p + q] = 2*pi*p/x_max;
        }
    }

    for (int p = M/2 + 1; p < M; p++) {
        for (int q = 0; q < N; q++) {
            kx[M*p + q] = 2*pi*(p - M)/x_max;
        }
    }

    return kx; 
}

double* ky2D_func(int M, int N, double y_max) {
    double* ky = (double*)malloc(sizeof(double) * M * N);

    for (int p = 0; p < M; p++) {
        for (int q = 0; q < N/2 + 1; q++) {
            ky[N*p + q] = 2*pi*q/y_max;
        }

        for (int q = N/2 + 1; q < N; q++) {
            ky[N*p + q] = 2*pi*(q - N)/y_max;
        }
    }

    return ky; 
}

int main() {
    int M = 5; int N = 5;
    double* kx2D = kx2D_func(M, N, 2*pi);
    double* ky2D = ky2D_func(M, N, 2*pi);
    double* kx1D = kx1D_func(M, 2*pi);

    printf("kx1D array: \n");
    for (int i = 0; i < M; i++) {
        printf("%f ", kx1D[i]);
    }
    printf("\n");

    printf("kx2D array: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", kx2D[i + M*j]);
        }
        printf("\n");
    }

    printf("ky2D array: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", ky2D[i + M*j]);
        }
        printf("\n");
    }

}