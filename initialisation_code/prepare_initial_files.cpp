#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <cublas.h>
#include <openacc.h> // used for OpenACC
#include <cmath>
#include <iostream>
#include <fstream>

#include "constants.h"
#include "declare_tables.h" //--> malloc_all

#include "initialisation_functions.h" //--> ini_phi, ini_ECM_lumen, ini_rho, ini_phin
#include "useful_functions.h" //--> check_nan, saving_out_functions, find_centres_cells, copy_cell_tables_1_by_1, compute_normal_grad_phi, boundary_conditions_cells, compute_normal_to_phi
                              // --> boundary_conditions_lumen_and_ECM
#include "cell_phase_functions.h" //--> reshaping_cells, reshaping_lumen_or_ECM, update_rho_to_shape, update_lumen_and_ECM

#define PI 3.14159265358979323846264338327

int main(int argc, char *argv[])
{


    srand(time(NULL));
    
    /* Option : If you want to measure the time */
    double timei;
    double timef;
    timei = time(NULL);
    double proci;
    double procf;
    /* end of the option */

    // other variables needed
    int nancount = 0;
    //declare FILES

    FILE *ini_phi_out_bef_reshaping=NULL, *ini_rho_out_bef_reshaping=NULL, *ini_phi_out=NULL, *ini_rho_out=NULL, *ini_lumen_out=NULL, *ini_ECM_out=NULL;
    char filename[64];
    
    //allocation of tables
    malloc_all();

    //initialisation of variables

    fprintf(stderr,"Ntotsave = %d \n",Ntotsave);
    
    fprintf(stderr,"I am done allocating. \n");

    // initialisation
    
    fprintf(stderr,"I start to initialize the variables.\n \n");

    ini_phi(phi);

    ini_ECM_lumen(lum, ECM, phi);

    for(int n=0;n<N_ini;n++)
    {   
        ini_rho(rho, phi, rho0, tol, n);
        ini_phin(phin,n);
    }

    fprintf(stderr,"I am done initializing.\n \n");
    proci = clock();

    nancount = check_nan(phi);
    if (nancount == 0)
    {fprintf(stderr,"No nans, before entering GPU.\n \n");}
    else
    {fprintf(stderr,"There are nans before entering the GPU.\n \n");}

    
/* enter in GPU device */

    fprintf(stderr,"GPUin\n");
    
    #pragma acc data copyin(phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phin[0:Ntot*N_s*N_s], rhon[0:Ntot*N_s*N_s], lum[0:N_s*N_s],ECM[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s]) copyout(phiout[0:Ntot*N_s*N_s*Ntotsave],rhoout[0:Ntot*N_s*N_s*Ntotsave], lumout[0:N_s*N_s*Ntotsave],ECMout[0:N_s*N_s*Ntotsave]) create(lumn[0:N_s*N_s], ECMn[0:N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],sum_s, psi[0:N_s*N_s], temp_phase[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], norm_grad_lum[0:N_s*N_s], J_x[0:Ntot*N_s*N_s], J_y[0:Ntot*N_s*N_s], hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s])
    {
    
    saving_out_functions(0, phi, rho, ECM, lum, phiout, rhoout, lumout, ECMout); 

    find_centres_cells(phi, xcenters, ycenters);

    fprintf(stderr,"copy_phi_to_phin\n");

    for (int n=0; n<Ntot; n++)
    {
    copy_cell_tables_1_by_1(n, phin, phi);
    }

    fprintf(stderr,"it = 0\n");
    int it = 0;
    fprintf(stderr,"compute_norm_grad_phi\n");
    compute_normal_grad_phi(norm_grad_phi, phi);
    fprintf(stderr,"boundary_conditions_cells\n");
    boundary_conditions_cells(norm_grad_phi);

    fprintf(stderr,"Prepare quantities to reshape cells.\n");
    compute_normal_to_phi(phi, norm_grad_phi, J_x, J_y);
    boundary_conditions_cells(J_x);
    boundary_conditions_cells(J_y);

    
    fprintf(stderr,"Reshape cells phase fields.\n");
    reshaping_cells(it, phi, phin, norm_grad_phi, J_x, J_y, xcenters, ycenters);
    fprintf(stderr,"Update the active gel density to the reshaped cells.\n");
    update_rho_to_shape(rho, phi, rho0, tol);

    update_lumen_and_ECM(lum,lumn,ECM,ECMn, hfunc_lum, hfunc_ECM, psi);
    boundary_conditions_lumen_and_ECM(lumn,ECMn);
    fprintf(stderr,"Reshape lumen and ECM phase fields.\n");
    reshaping_lumen_or_ECM(it,lum,lumn, 0.0);
    reshaping_lumen_or_ECM(it,ECM,ECMn, 1.0);

    saving_out_functions(nsaving, phi, rho, ECM, lum, phiout, rhoout, lumout, ECMout);
    }
    #pragma acc exit data delete(phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phin[0:Ntot*N_s*N_s], rhon[0:Ntot*N_s*N_s], lum[0:N_s*N_s],ECM[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s], lumn[0:N_s*N_s], ECMn[0:N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],sum_s, psi[0:N_s*N_s], temp_phase[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], J_x[0:Ntot*N_s*N_s], J_y[0:Ntot*N_s*N_s], hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s], norm_grad_lum[0:N_s*N_s])
    fprintf(stderr,"OUT OF GPU ! Let's save ! \n\n\n");

    fprintf(stderr,"Initial conditions every Nsaving! \n");
    for(int n=0;n<N_ini;n++){

    it = 0;

    snprintf(filename, sizeof(filename), "data_ini_to_plot/ini_cell_%d_before_reshaping.dat", n+1);
    ini_phi_out_bef_reshaping = fopen(filename, "w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
        fprintf(ini_phi_out_bef_reshaping,"%lg\n",phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
        fprintf(ini_phi_out_bef_reshaping,"\n");}
        fclose(ini_phi_out_bef_reshaping);
    
    
    snprintf(filename, sizeof(filename), "data_ini_to_plot/ini_rho_%d_before_reshaping.dat", n+1);
    ini_rho_out_bef_reshaping = fopen(filename, "w");
    
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_rho_out_bef_reshaping,"%lg\n",rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
    fprintf(ini_rho_out_bef_reshaping,"\n");}
    fclose(ini_rho_out_bef_reshaping);
    
    it = 1;


    snprintf(filename, sizeof(filename), "ini_cell_%d.bin", n+1);
    std::ofstream cell_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    cell_next.write(reinterpret_cast<char const*>(&phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]), sizeof(&phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]));
    }}
    cell_next.close();

    snprintf(filename, sizeof(filename),"ini_rho_%d.bin", n+1);
    std::ofstream rho_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    rho_next.write(reinterpret_cast<char const*>(&rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]), sizeof(&rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]));
    }}
    rho_next.close();

    
    snprintf(filename, sizeof(filename), "data_ini_to_plot/ini_cell_%d.dat", n+1);
    ini_phi_out = fopen(filename, "w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
        fprintf(ini_phi_out,"%lg\n",phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
        fprintf(ini_phi_out,"\n");}
        fclose(ini_phi_out);
    
    
    snprintf(filename, sizeof(filename), "data_ini_to_plot/ini_rho_%d.dat", n+1);
    ini_rho_out = fopen(filename, "w");
    
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_rho_out,"%lg\n",rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
    fprintf(ini_rho_out,"\n");}
    fclose(ini_rho_out);

    }

    it = 1;

    sprintf(filename,"ini_lumen.bin");    
    std::ofstream lumen_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    lumen_next.write(reinterpret_cast<char const*>(&lumout[it*N_s*N_s + i*N_s+j]), sizeof(&lumout[it*N_s*N_s + i*N_s+j]));
    }}
    lumen_next.close();
    
    sprintf(filename,"ini_ECM.bin");    
    std::ofstream ecm_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    ecm_next.write(reinterpret_cast<char const*>(&ECMout[it*N_s*N_s + i*N_s+j]), sizeof(&ECMout[it*N_s*N_s + i*N_s+j]));
    }}
    ecm_next.close();

    
    sprintf(filename,"data_ini_to_plot/ini_lumen.dat");
    ini_lumen_out = fopen(filename,"w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_lumen_out,"%g\n",lumout[it*N_s*N_s + i*N_s+j]);}
    fprintf(ini_lumen_out,"\n");}
    fclose(ini_lumen_out);
    
    sprintf(filename,"data_ini_to_plot/ini_ECM.dat");
    ini_ECM_out = fopen(filename,"w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_ECM_out,"%g\n",ECMout[it*N_s*N_s+i*N_s+j]);}
    fprintf(ini_ECM_out,"\n");}
    fclose(ini_ECM_out);

    fprintf(stderr,"I saved initial conditions ! Let's run the show ! \n\n\n");


    free(phi);
    free(phin);
    free(rho);
    free(rhon);
    free(J_x);
    free(J_y);
    free(prod_norm_delta);
    free(norm_grad_phi);
    free(norm_grad_lum);
    free(hfunc_cell);
    free(lum);
    free(lumn);
    free(ECM);
    free(ECMn);
    free(psi);
    free(hfunc_lum);
    free(hfunc_ECM);
    free(temp_phase);
    free(xcenters);
    free(ycenters);
    
    free(phiout);
    free(rhoout);
    free(lumout);
    free(ECMout);

    /* Option : If you want to measure the time */
    timef = time(NULL);
    procf = clock();
    fprintf(stderr, "run time %d [sec] (end %d - start %d)\n", (int)(timef-timei), timef, timei);
    fprintf(stderr, "proc time %g [sec]\n", (double)(procf-proci)/(double)CLOCKS_PER_SEC);
    /* end of the option */    

    return(0);


}
