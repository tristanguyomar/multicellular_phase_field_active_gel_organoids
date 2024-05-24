#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <openacc.h> // used for OpenACC

#include "constants.h"
#include "declare_tables.h"
#include "useful_functions.h"


extern void compute_prod_norm_delta(float *, float *, float *, int *, int *);

extern void compute_pi_xx(int, float *, float *, float *, float *, float *, float *, float*, int *, int*);
extern void compute_pi_yy(int, float *, float *, float *, float *, float *, float *, float*, int *, int *);
extern void compute_pi_xy(int, float *, float *, float *, float *, float *, float *, int *, int *);

//extern void compute_active_gel_parameters(float *, float *, float *, float *, float *, float*);

extern void randomize_rho(int, float *);

extern void update_rho_velocity(int, cufftComplex *, cufftComplex *, cufftComplex *, cufftComplex *, cufftComplex *, float *, int *, int *);
extern void create_deg_gen(int cell_number, cufftComplex *temp_fft, float *rho, float *prod_norm_delta, float *degrates, float *matgenrates, int *xcenters, int *ycenters);
extern void create_turnover_rates_tensions(float *degrates, float *matgenrates, float *a_tan_1, float *a_ort_1, float *viscosity, float *b, float *phi, float *lum, float *ECM, int *xcenters, int *ycenters);
extern void update_fft_rho(cufftComplex *, cufftComplex *, cufftComplex *, cufftComplex *);
extern void copy_to_rhon(int n, cufftComplex *temp_fft, float *rhon, int *xcenters, int *ycenters);
extern void load_fft_rho_v(int cell_number, cufftComplex *temp_fft, float *rho, cufftComplex *temp_fft_real, int *xcenters, int *ycenters);
extern void update_all_cortices(float *phi, float *rho, float *rhon, int *xcenters, int *ycenters, float *degrates, float *matgenrates, float *a_tan_1, float *a_ort_1, float* viscosity, float *b, float *norm_grad_phi, float *prod_norm_delta, cufftHandle planf, cufftHandle planr, cufftComplex *temp_fft, cufftComplex *temp_fft_real, cufftComplex *fft_rho, cufftComplex *fft_pi_xx, cufftComplex *fft_pi_xy, cufftComplex *fft_pi_yy, cufftComplex *fft_vx, cufftComplex *fft_vy, cufftComplex *fft_react, cufftComplex *fft_rho_vx, cufftComplex *fft_rho_vy);