#include "constants.h"
#include "declare_tables.h"
#include "declare_fft_tables.h"
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>


// Cuda FFT functions
//declare fft variables
cufftComplex * __restrict temp_fft;
cufftComplex * __restrict temp_fft_real;
cufftComplex * __restrict fft_pi_xx;
cufftComplex * __restrict fft_pi_yy;
cufftComplex * __restrict fft_pi_xy;
cufftComplex * __restrict fft_rho_vx;
cufftComplex * __restrict fft_rho_vy;
cufftComplex * __restrict fft_vx;
cufftComplex * __restrict fft_vy;
cufftComplex * __restrict fft_rho;
cufftComplex * __restrict fft_react;

cufftHandle planf, planr;   // This defines the "plan" variable for CUDA FFT, which will be defined later specifically at the start of the main code 


// three dimensional tables
void ini_fft_all()
{
    // declare Cuda FFT complex variables
    temp_fft = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_pi_xx = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_pi_yy = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_pi_xy = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_rho_vx = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_rho_vy = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_vx = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_vy = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    temp_fft_real = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_rho = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));
    fft_react = (cufftComplex *)malloc((2*zone_size-2)*(2*zone_size-2)* sizeof(cufftComplex));

}

