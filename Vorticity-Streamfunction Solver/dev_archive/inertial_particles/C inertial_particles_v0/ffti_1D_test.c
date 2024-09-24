# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <complex.h>
# include <fftw3.h>

// gcc ffti_1D_test.c -o ffti_1D_test.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./ffti_1D_test.out

const double pi = 3.14159265358979323846;

double* kx1D_func(int M, double x_max) {
    double* kx = (double*)malloc(sizeof(double) * M);
    
    for (int p = 0; p <= M/2; p++) {
        kx[p] = 2*pi*p/x_max;
    }

    for (int p = M/2 + 1; p < M; p++) {
        kx[p] = 2*pi*(p - M)/x_max;
    }

    return kx;
}

double inter_1D(int M, double kx1D[], double _Complex U[], double x_pos) {
    double _Complex value = 0;
    
    for (int p = 0; p < M; p++) {
        value += U[p] * cexp(I * kx1D[p] * x_pos);
    }
    value /= M;
    return value;
}

int main() {
    int M = 9;
    double x_max = 2*pi;

    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M);

    fftw_plan plan = fftw_plan_dft_1d(M, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    
    printf("\nOriginal Data:\n");
    for (int i = 0; i < M; i++) {
        double x = i*x_max/M;
        in[i] = exp(sin(x));
        printf("%f + %fi\n", crealf(in[i]), cimagf(in[i]));
    }
    
    fftw_execute(plan);

    double* kx = kx1D_func(M, x_max);

    printf("\nFFT Result:\n");
    for (int i = 0; i < M; i++) {
        printf("%f + %fi\n", crealf(out[i]), cimagf(out[i]));
    }
    printf("\n");

    double x_int = 1.3512;
    double _Complex y_int = inter_1D(M, kx, out, x_int);

    printf("Interpolation Error at %f: %.10f\n", x_int, crealf(y_int - exp(sin(x_int))));

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}