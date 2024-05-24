#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <openacc.h> // used for OpenACC

#include "constants.h"
#include "declare_tables.h"
#include "useful_functions.h"

#include <iostream>
#include <fstream>
#include <unistd.h>


extern void update_rho_to_shape(float *rho, float *phi, float rho0, float tol);
extern void update_all_cells_only_target_volume_and_adhesion(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *vol_diff , float *psi, int *xcenters, int *ycenters);
extern void update_all_cells(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *vol_diff , float *psi, int *xcenters, int *ycenters, float *hfunc_lum, float *hfunc_ECM);
extern void update_cells(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *f0 , float *psi, float *hfunc_lum, float *hfunc_ECM);
extern void update_lumen_and_ECM(float *lum, float *lumn, float *ECM, float *ECMn, float *hfunc_lum, float *hfunc_ECM, float *psi);

extern void reshaping_cells(int, float *, float *, float *, float *, float *, int *,int *);
extern void reshaping_cells_all_grid(int, float *, float *, float *, float *);

extern void reshaping_lumen_or_ECM(int, float *, float *, float);

// extern void copy_phase(float **, float **);

// extern void reshaping_lumen(int, float **, float **);
// extern void reshaping_ECM(int, float **, float **);

// extern void update_lumen(float **, float **, float **, float **);

// extern void update_ECM(float **, float **, float **, float **);


extern void print_mem(int, char*);