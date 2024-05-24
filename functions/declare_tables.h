#include <stdio.h>
#include <stdlib.h>

// declare used functions
extern float* malloc_1_dim_float(int);
extern void malloc_all(void);
// declare cell phase variables and lumen and ECM 

extern float * restrict phi;
extern float * __restrict phin;
extern float * __restrict rho;
extern float * __restrict rho_to_compare;
extern float * __restrict rhon;
extern float * __restrict lum;
extern float * __restrict ECM;
extern float * __restrict lumn;
extern float * __restrict ECMn;

extern float * __restrict psi;
extern float * __restrict temp_phase; 

// geometrical functions to characterize cells, lumen, ECM 
extern float * __restrict  hfunc_cell;
extern float * __restrict hfunc_lum;
extern float * __restrict hfunc_ECM;

// needed quantities
extern float * __restrict norm_grad_phi;
extern float * __restrict prod_norm_delta;
extern float * __restrict norm_grad_lum;
extern float * __restrict norm_grad_ecm;


// active gel parameter 
extern float * __restrict a_tan_1;
extern float * __restrict a_ort_1;


extern float * __restrict b;
extern float * __restrict viscosity;



// stress tensor for the gel
extern float * __restrict pi_xx;
extern float * __restrict pi_yy;
extern float * __restrict pi_xy;

// readouts and saving
extern float * __restrict phiout;
extern float * __restrict rhoout;
extern float * __restrict lumout;
extern float * __restrict ECMout;
extern float * __restrict Jxout;
extern float * __restrict Jyout;

extern float * __restrict vol_out;

// reshaping process
extern float * __restrict J_x;
extern float * __restrict J_y;

extern int * __restrict xcenters;
extern int * __restrict ycenters;

extern float * __restrict matgenrates;
extern float * __restrict degrates;

extern float * __restrict matout;
extern float * __restrict degout;

