// gcc test.c -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define MAX(a,b) ( (a) > (b) ? (a) : (b))

#define ck_alloc(e) {if(!(e)){fprintf(stderr, "%s:%d: could not allocate memory for \'%s\'.\nAbort.\n",__FILE__,__LINE__,#e); exit(8);}}

int main(int argc, char *argv[])
{
    int im=32;
    int jm=32;
    //int jm2=jm/2;
    //int im2p=im/2+1;
    int jm2p=jm/2+1;

    double *in1=(double *)malloc(im*jm*sizeof(double));
    ck_alloc(in1);

    fftw_complex *out1=(fftw_complex *)fftw_malloc(im*jm2p*sizeof(fftw_complex));
    ck_alloc(out1);
    
    fftw_plan plan1=fftw_plan_dft_r2c_2d(im,jm,in1,out1,FFTW_MEASURE);

    fftw_destroy_plan(plan1);
    
    fftw_free(in1);
    fftw_free(out1);

    fftw_cleanup();
    
    printf("hello\n");
    
    return 0;
}