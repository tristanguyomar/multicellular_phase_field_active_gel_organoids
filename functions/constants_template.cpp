#include "constants.h"
#include <math.h>

int threshold_reshaping_steps = 100000;
int riprint = 100000; // print infos during relaxation of the profile
int rdev_threshold = 5.0;
int rdev_ini = 2*rdev_threshold;
// simulation parameters
int Nsteps = 11; // number of integration steps
// saving and printing parameters
int ncontinue = 0;
int ndisplay = 5;
int nsaving = 10; // saving every nsaving steps
int Ntotsave = (1 + (int)((Nsteps-1)/nsaving));

// time and space resolution parameters
float dx = 0.01;
float dt = 0.002;

// Gridsize parameters
const int N_s = 1200; // size of the grid
const int L = 120;

// cell size parameters
float R0 = 1.0;
float R = R0;
float s0 = 3.14159265359*R0*R0+0.01;
float Rint = 1.0;
float Rout = sqrt(Rint*Rint+N_ini*s0/3.14159265359);

float sigma = 0.2;

// number of cells parameters
int N_ini = {{ncells}}; // number of initial cells
const int Ntot = N_ini; // total number of cells
int N_current = N_ini; // current number of cells 

int zone_size = 200;

// activegel phase field coupling

float coupling_PF_AG = 1.0;

// active gel activation time parameters
int random_ontime = Nsteps+1; // time of randomization of the active gel density 
float seuil_gel = 0.1; // during randomization, prevents the background to get noisy (seuil (french) == threshold (english)) 
int gen_ontime = 0; // time at which generation/degeneration of the gel is turned on
int active_gel_ontime = 0; // time at which the activity gel is turned on
int contract_ontime = 0; // time at which parameters of the gel can be changed

// active gel parameters

float epsilon_ap = {{epsilon_ap}}; // to convert a_tn from a_ort
float epsilon_bas = {{epsilon_bas}}; // to convert a_tn from a_ort
float epsilon_lat = {{epsilon_lat}}; // to convert a_tn from a_ort

float a_ort_lateral = {{a_ort_lat}};
float a_ort_apical = {{a_ort_ap}};
float a_ort_basal = {{a_ort_bas}};

float a_tan_lateral = a_ort_lateral*epsilon_lat;
float a_tan_apical = a_ort_apical*epsilon_ap;
float a_tan_basal = a_ort_basal*epsilon_bas;

float b_ap = {{b_ap}}; // (everywhere) 
float b_bas = {{b_bas}}; // (everywhere)
float b_lat = {{b_lat}}; // (everywhere) 

float matgenrate_apical = {{gen_ap}}; //generation after gen_ontime
float degrate_apical = {{deg_ap}};// degeneration after gen_ontime

float matgenrate_lateral = 1.0; //generation after gen_ontime
float degrate_lateral = 100.0;// degeneration after gen_ontime
float matgenrate_basal = matgenrate_apical; //generation after gen_ontime
float degrate_basal = degrate_apical;// degeneration after gen_ontime

float friction = {{friction}};
float viscosity_ap = {{viscosity_ap}};
float viscosity_bas = {{viscosity_bas}};
float viscosity_lat = {{viscosity_lat}};

//phase field parameters

float tau = 1.0;
float alpha = 1.0;
float beta = 1.0;
float eta = {{eta}};
float gamma_cell = eta*1.5;

// lumen parameters
float tau_lum = 1.0;
float beta_lum_cells = 1.0;
float eta_lum_cells = {{eta_lum_cells}};
float gamma_lum_cells = eta_lum_cells*1.1;
float xi_lum = {{xi_lumen}};

// ECM parameters
float alpha_ECM = {{alpha_ECM}};
float tau_ECM = 1.0;
float beta_ECM_cells = 1.0;
float eta_ECM_cells = {{eta_ecm_cells}};
float gamma_ECM_cells = eta_ECM_cells*1.1;
float xi_ECM = 0.0;

// interaction parameters
float beta_lum_ECM = 1.0;

//initial conditions
float wi = 0.07; //width of the phase field interface
float rho0 = 0.001;
float tol = 0.01;

// reshaping algorithm parameters
int ri;
int rimax = 1000000;
float rdev;
float rtol = 0.001;
float rdt = 0.001;

// to communicate during GPU computation time
float ans =0.0;
float sum_s=0.0;

// working indices

//indices needed
int it,k,l,ir,jr,c;
int nelse; //to caracterize the other cells
int ip,im,jp,jm;

