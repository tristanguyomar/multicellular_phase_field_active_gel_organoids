#include <math.h>

// simulation parameters
extern int Nsteps; // number of integration steps
extern int zone_size;
extern int threshold_reshaping_steps;
extern int rdev_ini;
extern int rdev_threshold;

// saving and printing parameters
extern int ncontinue;
extern int ndisplay;
extern int nsaving; // saving every nsaving steps
extern int riprint; // prextern int infos during relaxation of the profile
extern int Ntotsave;

// time and space resolution parameters
extern float dt;
extern float dx;

// Gridsize parameters
extern const int N_s;
extern const int L;

// cell size parameters
extern float R0;
extern float R;
extern float s0;
extern float Rint;
extern float Rout;

extern float  sigma;

// number of cells parameters
extern int N_ini; // number of initial cells
extern const int Ntot; // total number of cells
extern int N_current; // current number of cells 

extern float coupling_PF_AG; // activegel phase field coupling

// active gel activation time parameters
extern int random_ontime; // time of randomization of the active gel density 
extern float seuil_gel; // during randomization, prevents the background to get noisy (seuil (french) =threshold (english)) 
extern int gen_ontime; // time at which generation/degeneration of the gel is turned on
extern int active_gel_ontime; // time at which the activity gel is turned on
extern int contract_ontime; // time at which parameters of the gel can be changed

//active gel parameters "active_stress(rho) = - a*rho^3+b*rho^4"

extern float b_ap; // (everywhere) //updated at contract_ontime
extern float b_bas; // (everywhere) //updated at contract_ontime
extern float b_lat; // (everywhere) //updated at contract_ontime

extern float a_tan_lateral;
extern float a_tan_apical;
extern float a_tan_basal;

extern float a_ort_lateral;
extern float a_ort_apical;
extern float a_ort_basal;


extern float matgenrate_lateral; //generation after gen_ontime
extern float degrate_lateral; // degeneration after gen_ontime
extern float matgenrate_apical; //generation after gen_ontime
extern float degrate_apical; // degeneration after gen_ontime
extern float matgenrate_basal; //generation after gen_ontime
extern float degrate_basal; // degeneration after gen_ontime


extern float friction;
extern float viscosity_ap;
extern float viscosity_bas;
extern float viscosity_lat;

//phase field parameters
extern float  D0;
extern float  tau;
extern float  alpha;
extern float  beta;
extern float  eta;
extern float gamma_cell;

// lumen parameters
extern float  D_lum;
extern float  tau_lum;
extern float  beta_lum_cells;
extern float eta_lum_cells;
extern float gamma_lum_cells;
extern float xi_lum;

// lumen parameters
extern float alpha_ECM;
extern float  D_ECM;
extern float  tau_ECM;
extern float  beta_ECM_cells;
extern float eta_ECM_cells;
extern float gamma_ECM_cells;
extern float xi_ECM;

// interaction parameters
extern float beta_lum_ECM;

//initial conditions
extern float  wi; //width of the phase field interface
extern float  rho0;
extern float  tol;

// reshaping algorithm parameters
extern int ri;
extern int rimax;
extern float rdev;
extern float rtol;
extern float rdt;


// to communicate during GPU computation time
extern float  ans;
extern float  sum_s;

// working indices
extern int i,j,it,n,k,l,ir,jr,c;
extern int nelse; //to caracterize the other cells
extern int ip,im,jp,jm;
