#include <stdio.h>
#include <stdlib.h>
#include "declare_tables.h"
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>


extern void ini_fft_all(void);

// Cuda FFT functions
//declare fft variables
extern cufftComplex * restrict temp_fft;
extern cufftComplex * restrict temp_fft_real;
extern cufftComplex * restrict fft_pi_xx;
extern cufftComplex * restrict fft_pi_yy;
extern cufftComplex * restrict fft_pi_xy;
extern cufftComplex * restrict fft_rho_vx;
extern cufftComplex * restrict fft_rho_vy;
extern cufftComplex * restrict fft_vx;
extern cufftComplex * restrict fft_vy;
extern cufftComplex * restrict fft_rho;
extern cufftComplex * restrict fft_react;

extern cufftHandle planf, planr;   
