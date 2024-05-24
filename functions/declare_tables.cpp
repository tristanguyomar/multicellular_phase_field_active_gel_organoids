#include "constants.h"
#include "declare_tables.h"


float* malloc_1_dim_float(int first_dim)
{
    float *array;

    array = (float *)malloc(first_dim*sizeof(float));
    return(array);
}

int* malloc_1_dim_int(int first_dim)
{
    int *array;

    array = (int *)malloc(first_dim*sizeof(int));
    return(array);
}

// definition of variables


// define cell phase variables and lumen and ECM 

float * __restrict phi;
float * __restrict phin;
float * __restrict rho;
float * __restrict rho_to_compare;
float * __restrict rhon;
float * __restrict lum;
float * __restrict ECM;
float * __restrict lumn;
float * __restrict ECMn;

float * __restrict  psi;
float * __restrict  temp_phase; 

// geometrical functions to characterize cells, lumen, ECM 
float * __restrict  hfunc_cell;
float * __restrict  hfunc_lum;
float * __restrict  hfunc_ECM;

// needed quantities
float * __restrict  norm_grad_phi;
float * __restrict  prod_norm_delta;
float * __restrict norm_grad_lum;
float * __restrict norm_grad_ecm;

// active gel parameter 
float * __restrict a_tan_1;
float * __restrict a_ort_1;

float * __restrict b;
float * __restrict viscosity;


// stress tensor for the gel
float * __restrict pi_xx;
float * __restrict pi_yy;
float * __restrict pi_xy;

// readouts and saving
float * __restrict phiout;
float * __restrict rhoout;
float * __restrict lumout;
float * __restrict ECMout;
float * __restrict Jxout;
float * __restrict Jyout;

float * __restrict vol_out;

// reshaping process
float * __restrict  J_x;
float * __restrict  J_y;

int *__restrict xcenters;
int *__restrict ycenters;

// turnover rates for the gel

float * __restrict matgenrates;
float * __restrict degrates;

float * __restrict matout;
float * __restrict degout;

// three dimensional tables
void malloc_all()
{
    phiout = malloc_1_dim_float(Ntot*N_s*N_s*Ntotsave);
    rhoout = malloc_1_dim_float(Ntot*N_s*N_s*Ntotsave);
    lumout = malloc_1_dim_float(N_s*N_s*Ntotsave);
    ECMout = malloc_1_dim_float(N_s*N_s*Ntotsave);
    Jxout = malloc_1_dim_float(N_s*N_s*Ntotsave*Ntot);
    Jyout = malloc_1_dim_float(N_s*N_s*Ntotsave*Ntot);
    matout = malloc_1_dim_float(N_s*N_s*Ntotsave*Ntot);
    degout = malloc_1_dim_float(N_s*N_s*Ntotsave*Ntot);
    
    
    phi = malloc_1_dim_float(Ntot*N_s*N_s);
    phin = malloc_1_dim_float(Ntot*N_s*N_s);
    rho = malloc_1_dim_float(Ntot*N_s*N_s);
    rho_to_compare = malloc_1_dim_float(Ntot*N_s*N_s);
    rhon = malloc_1_dim_float(Ntot*N_s*N_s);
    hfunc_cell = malloc_1_dim_float(Ntot*N_s*N_s);
    norm_grad_phi = malloc_1_dim_float(Ntot*N_s*N_s);
    prod_norm_delta = malloc_1_dim_float(Ntot*N_s*N_s);
    norm_grad_lum = malloc_1_dim_float(N_s*N_s);
    norm_grad_ecm = malloc_1_dim_float(N_s*N_s);
    J_x = malloc_1_dim_float(Ntot*N_s*N_s);
    J_y = malloc_1_dim_float(Ntot*N_s*N_s);

    // two dimensional tables

    psi = malloc_1_dim_float(N_s*N_s);
    temp_phase = malloc_1_dim_float(Ntot*N_s*N_s);

    lum = malloc_1_dim_float(N_s*N_s);
    lumn = malloc_1_dim_float(N_s*N_s);

    ECM = malloc_1_dim_float(N_s*N_s);
    ECMn = malloc_1_dim_float(N_s*N_s);

    hfunc_lum = malloc_1_dim_float(N_s*N_s);
    hfunc_ECM = malloc_1_dim_float(N_s*N_s);

    a_tan_1 = malloc_1_dim_float(Ntot*N_s*N_s);
    a_ort_1 = malloc_1_dim_float(Ntot*N_s*N_s);
    
    viscosity = malloc_1_dim_float(Ntot*N_s*N_s);
    b = malloc_1_dim_float(Ntot*N_s*N_s);

    pi_xx = malloc_1_dim_float(Ntot*N_s*N_s);
    pi_yy = malloc_1_dim_float(Ntot*N_s*N_s);
    pi_xy = malloc_1_dim_float(Ntot*N_s*N_s);

    xcenters = malloc_1_dim_int(Ntot);
    ycenters = malloc_1_dim_int(Ntot);

    matgenrates= malloc_1_dim_float(Ntot*N_s*N_s);
    degrates = malloc_1_dim_float(Ntot*N_s*N_s);
}

