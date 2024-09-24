# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <complex.h>
# include <fftw3.h>
const double pi = 3.14159265358979323846;

// the goal of this script is to simulate inertial particles in unperturbed Taylor-Greene flow, and output position in readable fashion for Python

// gcc inertial_particles.c -o inertial_particles.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3
// ./inertial_particles.out > inertial_particles.csv

const int M = 256;
const int N = 256;
const double nu = 9e-4;
const int beta = 1;
const double t_end = 100;
const double tau = 0;
const int np = 1;

double* kp2d_func(int M, int N, double x_max) {
    double* kp = (double*)malloc(sizeof(double) * M * N);

    for (int p = 0; p < M/2 + 1; p++) {
        for (int q = 0; q < N; q++) {
            kp[M*p + q] = 2*pi*p/x_max;
        }
    }

    for (int p = M/2 + 1; p < M; p++) {
        for (int q = 0; q < N; q++) {
            kp[M*p + q] = 2*pi*(p - M)/x_max;
        }
    }

    return kp; 
}

double* kq2d_func(int M, int N, double y_max) {
    double* kq = (double*)malloc(sizeof(double) * M * N);

    for (int p = 0; p < M; p++) {
        for (int q = 0; q < N/2 + 1; q++) {
            kq[N*p + q] = 2*pi*q/y_max;
        }

        for (int q = N/2 + 1; q < N; q++) {
            kq[N*p + q] = 2*pi*(q - N)/y_max;
        }
    }
    return kq; 
}

double inter_2d(int M, int N, double kp2d[], double kq2d[], double _Complex U[], double x_pos, double y_pos) {
    double _Complex value = 0;
    double _Complex value_temp = 0;

    for (int p = 0; p < M; p++) {
        value_temp = 0;
        for (int q = 0; q < N; q++) {
            value_temp += U[M*p + q] * cexp(I * kq2d[M*p + q] * y_pos);
        }
        value_temp /= N;
        value += value_temp * cexp(I * kp2d[M*p] * x_pos);
    }
    value /= M;
    return value;
}

double mod2pi(double x) {
    while (x < 0 || 2*pi <= x) {
        if (x < 0) {x += 2*pi; }
        if (x >= 2*pi) {x -= 2*pi; }
    }
    return x;
}

int main() {
    // initialize main variables
    double* kp = (double*) malloc(sizeof(double) * M * N);
    double* kq = (double*) malloc(sizeof(double) * M * N);
    kp = kp2d_func(M, N, 2*pi);
    kq = kp2d_func(M, N, 2*pi);

    fftw_complex* u = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);
    fftw_complex* v = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);
    fftw_complex* U = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);
    fftw_complex* V = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M * N);

    for (int i = 0; i < M; i++) {
        double x = 2*pi*i/M;
        for (int j = 0; j < N; j++) {
            double y = 2*pi*j/N;

            u[M*i + j] = + cos(beta*x) * sin(beta*y);
            v[M*i + j] = - sin(beta*x) * cos(beta*y);   
        }
    }

    fftw_plan plan_u2U = fftw_plan_dft_2d(M, N, u, U, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_v2V = fftw_plan_dft_2d(M, N, v, V, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_U2u = fftw_plan_dft_2d(M, N, U, u, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan plan_V2v = fftw_plan_dft_2d(M, N, V, v, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_u2U);
    fftw_execute(plan_v2V);

    // initialize particle variables
    double* xp = (double*) malloc(sizeof(double) * 2 * np);
    double* vp = (double*) malloc(sizeof(double) * 2 * np);
    double* up = (double*) malloc(sizeof(double) * 2 * np);
    double* ap = (double*) malloc(sizeof(double) * 2 * np);

    double* xp_star = (double*) malloc(sizeof(double) * 2 * np);
    double* vp_star = (double*) malloc(sizeof(double) * 2 * np);
    double* up_star = (double*) malloc(sizeof(double) * 2 * np);
    double* up_next = (double*) malloc(sizeof(double) * 2 * np);
    double* ap_next = (double*) malloc(sizeof(double) * 2 * np);

    for (int n = 0; n < np; n++) {
        xp[n] = 2*pi*rand()/RAND_MAX;
        xp[n+1] = 2*pi*rand()/RAND_MAX;
        
        vp[n] = 0;
        vp[n+1] = 0;
    }

    fprintf(stdout, "t,");
    for (int n = 0; n < np; n++) {
        fprintf(stdout, "x%d,y%d,", n, n);
    }
    fprintf(stdout, "\n");

    double t = 0;
    double dt = 0.1;
    while (t < t_end) {
        // update velocity field
        for (int i = 0; i < M; i++) {
            double x = 2*pi*i/M;
            for (int j = 0; j < N; j++) {
                U[M*i + j] *= exp(-nu * dt);
                V[M*i + j] *= exp(-nu * dt);
            }
        }

        fprintf(stdout, "%.10f,", t);
        for (int n = 0; n < np; n++) {
            if (tau == 0) {
                vp[n] = creal(inter_2d(M, N, kp, kq, U, xp[n], xp[n+1]));
                vp[n+1] = creal(inter_2d(M, N, kp, kq, V, xp[n], xp[n+1]));

                xp_star[n] = xp[n] + dt*vp[n];
                xp_star[n+1] = xp[n+1] + dt*vp[n+1];
                
                vp_star[n] = creal(inter_2d(M, N, kp, kq, U, xp_star[n], xp_star[n+1]));
                vp_star[n+1] = creal(inter_2d(M, N, kp, kq, V, xp_star[n], xp_star[n+1]));

                xp[n] = xp[n] + dt * (vp[n] + vp_star[n])/2;
                xp[n+1] = xp[n+1] + dt * (vp[n+1] + vp_star[n+1])/2;
            }

            if (tau != 0) {
                // predictor substep
                xp_star[n] = xp[n] + dt * vp[n];
                xp_star[n+1] = xp[n+1] + dt * vp[n+1];
                
                up_star[n] = exp(-nu * dt) * creal(inter_2d(M, N, kp, kq, U, xp[n], xp[n+1]));
                up_star[n+1] = exp(-nu * dt) * creal(inter_2d(M, N, kp, kq, V, xp[n], xp[n+1]));

                ap[n] = -nu * exp(nu * dt) * up_star[n];
                ap[n+1] = -nu * exp(nu * dt) * up_star[n+1];

                vp_star[n] = (vp[n] + dt/tau * (up_star[n] + 3*tau*ap[n]))/(1 + dt/tau);
                vp_star[n+1] = (vp[n+1] + dt/tau * (up_star[n+1] + 3*tau*ap[n+1]))/(1 + dt/tau);
                
                // corrector substep
                xp[n] += dt/2 * (vp[n] + vp_star[n]);
                xp[n+1] += dt/2 * (vp[n+1] + vp_star[n+1]);

                up_next[n] = exp(-nu * dt) * creal(inter_2d(M, N, kp, kq, U, xp[n], xp[n+1]));
                up_next[n+1] = exp(-nu * dt) * creal(inter_2d(M, N, kp, kq, V, xp[n], xp[n+1]));

                ap_next[n] = -nu * up[n];
                ap_next[n+1] = -nu * up[n+1];

                vp[n] = 2*tau/(2*tau + dt) * (vp[n] + dt/2 * (3 * (ap[n] + ap_next[n]) + (up[n] + up_next[n] - vp[n])/tau));
                vp[n] = 2*tau/(2*tau + dt) * (vp[n+1] + dt/2 * (3 * (ap[n+1] + ap_next[n+1]) + (up[n+1] + up_next[n+1] - vp[n+1])/tau));

                // reassign temporary variables
                up[n] = up_next[n];
                up[n+1] = up_next[n+1];
            }

            xp[n] = mod2pi(xp[n]);
            xp[n+1] = mod2pi(xp[n+1]);

            fprintf(stdout, "%.10f,%.10f,", xp[n], xp[n+1]);
        }
        
        fprintf(stdout, "\n");
        fprintf(stderr, "t = %.10f\r", t);

        t += dt;
    }

    free(u); free(v);
    free(kp); free(kq);

    free(xp); free(vp); free(up); free(ap);
    free(xp_star); free(vp_star); free(up_star); free(up_next); free(ap_next);
}