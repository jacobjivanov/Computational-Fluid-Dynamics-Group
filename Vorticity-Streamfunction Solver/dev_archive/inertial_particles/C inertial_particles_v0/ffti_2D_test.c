# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <complex.h>
# include <fftw3.h>

// gcc ffti_2D_test.c -o ffti_2D_test.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./ffti_2D_test.out

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

double inter_1D(int M, double kx1D[], double _Complex U[], double x_pos) {
    double _Complex value = 0;
    
    for (int p = 0; p < M; p++) {
        value += U[p] * cexp(I * kx1D[p] * x_pos);
    }
    value /= M;
    return value;
}

double inter_2D(int M, int N, double kx2D[], double ky2D[], double _Complex U[], double x_pos, double y_pos) {
    double _Complex value = 0;
    double _Complex value_temp = 0;

    for (int p = 0; p < M; p++) {
        value_temp = 0;
        for (int q = 0; q < N; q++) {
            value_temp += U[M*p + q] * cexp(I * ky2D[M*p + q] * y_pos);
        }
        value_temp /= N;
        value += value_temp * cexp(I * kx2D[M*p] * x_pos);
    }
    value /= M;
    return value;
}

double f(double x, double y) {
    return exp(sin(x) + sin(y));
}

int main() {
    int M = 13; int N = M;
    double x_max = 2*pi; double y_max = 2*pi;
    
    fftw_complex* u = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);
    fftw_complex* U = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);

    fftw_plan plan_u2U = fftw_plan_dft_2d(M, N, u, U, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_U2u = fftw_plan_dft_2d(M, N, U, u, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    double x; double y;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            x = i*x_max/M; y = j*y_max/N;
            u[M*i + j] = f(x, y);
            // printf("%.5f | ", crealf(u[M*i + j]));
        }
        // printf("\n");
    }
    
    fftw_execute(plan_u2U);

    // printf("\nFFT Result:\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {

    //         printf("%.5f + %.5fi | ", crealf(U[M*i + j]), cimagf(U[M*i + j]));
    //     }
    //     printf("\n");
    // }

    double* kx2D = kx2D_func(M, N, x_max);
    double* ky2D = ky2D_func(M, N, y_max);
    double x_int = 1.2903857; double y_int = 2.934857;
    double _Complex u_int = inter_2D(M, N, kx2D, ky2D, U, x_int, y_int);

    printf("\nInterpolation at (%.5f, %.5f): %.5f + %.5fi\n", x_int, y_int, creal(u_int), cimag(u_int));
    printf("Interpolation Error: %.5f + %.5fi\n", creal(u_int - f(x_int, y_int)), cimag(u_int - f(x_int, y_int)));

    fftw_destroy_plan(plan_u2U);
    fftw_destroy_plan(plan_U2u);
    fftw_free(u);
    fftw_free(U);
}