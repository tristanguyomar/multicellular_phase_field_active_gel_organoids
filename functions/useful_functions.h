#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "declare_tables.h"
//#include "declare_fft_tables.h"
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <cmath>

extern int check_nan(float *);

extern int check_initial_condition(float *, float *);

extern void saving_turnover_rates_zones(int it, float *matgenrates, float *degrates, float *matout, float *degout);
extern void saving_out_functions(int it, float *phi, float *rho, float *ECM, float *lum, float *phiout, float *rhoout, float *lumout, float *ECMout);
extern void saving_out_functions_with_normals(int it, float *phi, float *rho, float *ECM, float *lum, float *phiout, float *rhoout, float *lumout, float *ECMout, float *J_x, float *J_y, float *Jxout, float *Jyout);

extern void psi_to_zero(float *);
extern void psi_to_zero_kernels(float *);
extern void compute_hfunc_3d_tables_kernels(float *, float *);
extern void compute_hfunc_3d_tables_kernels_loop_n(float *, float *, int *, int*);

extern void compute_hfunc_2d_tables_kernels(float *, float *);
extern void compute_all_hfunc_2d_tables_kernels(float *, float *, float *, float *);
extern void compute_psi_kernels(float *, float *);

extern void compute_normal_grad_phi(float *, float *);
extern void compute_normal_to_phi(float *phi, float *norm_grad_phi, float *J_x, float *J_y);
extern void compute_prod_norm_delta(float *, float *, float *);
extern void sum_s_kernels(float *, float *,float*);
extern void find_centres_cells(float *phi, int *xcenters, int *ycenters);

extern void boundary_conditions_cells(float *);
extern void boundary_conditions_dirichlet(float * lum, float side_value);
extern void boundary_conditions_lumen_and_ECM(float *, float*);
extern void boundary_conditions_periodic(float*);
extern void boundary_conditions_Jx(float *);
extern void boundary_conditions_pi(float *, float *, float *);
extern void boundary_conditions_lumen_Jy(float *);

extern void load_temp_fft(int cell_number, float *table, cufftComplex *temp_fft, int *xcenters, int *ycenters);
extern void load_temp_fft_rho(int cell_number, float *rho, cufftComplex *temp_fft, cufftComplex *temp_fft_real, int *xcenters, int *ycenters);
extern void compute_fft_forward(cufftHandle , cufftComplex *, cufftComplex *);
extern void compute_fft_backward(cufftHandle, cufftComplex *, cufftComplex *);

extern void copy_cell_tables(float *, float *);
extern void copy_cell_tables_1_by_1(int n, float *phase, float *phasen);
extern void copy_tables_ECM_or_lumen(float *phase, float *phasen);
