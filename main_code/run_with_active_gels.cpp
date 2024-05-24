#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <openacc.h> // used for OpenACC
#include <cmath>
#include <iostream>
#include <fstream>

#include "constants.h"
#include "declare_tables.h" //--> malloc_all
#include "declare_fft_tables.h" //--> ini_fft_all

#include "initialisation_functions.h"
#include "useful_functions.h"
#include "rho_functions.h"
#include "cell_phase_functions.h"


#define PI 3.14159265358979323846264338327

int main(int argc, char *argv[])
{
    srand(time(NULL));
    
    /* Option : If you want to measure the time */
    float timei;
    float timef;
    timei = time(NULL);
    float proci;
    float procf;
    /* end of the option */

    // other variables needed
    float * s; //volume of the cells
    float * f0; //volume growth factor for each cell

    //declare FILES

    FILE *ini_phi_out=NULL, *ini_rho_out=NULL, *ini_lumen_out=NULL, *ini_ECM_out=NULL;
    char filename[64];
    
    //allocation of tables
    malloc_all();

    //allocation of all fft tables
    ini_fft_all();

    //initialisation of variables

    fprintf(stderr,"Ntotsave = %d \n",Ntotsave);
    
    s = (float *)malloc(Ntot*sizeof(float));
    f0 = (float *)malloc(Ntot*sizeof(float));

    fprintf(stderr,"I am done allocating. \n");

    // initialisation
    
    
    fprintf(stderr,"I start to initialize the variables from files.\n \n");

    ini_phi_files(phi);
    ini_ECM_files(ECM);
    ini_lumen_files(lum);
    ini_rho_files(rho);

    for(int n=0;n<N_ini;n++)
    {
        ini_phin(phin,n);
        ini_rhon(rhon,n);
        ini_phin(hfunc_cell, n);
    }

    fprintf(stderr,"I am done initializing.\n \n");
    proci = clock();

    /* enter in GPU device */
    fprintf(stderr,"GPUin\n");

    #pragma acc data copyin(phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phin[0:Ntot*N_s*N_s], rhon[0:Ntot*N_s*N_s], lum[0:N_s*N_s],ECM[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s]) copyout(degout[0:N_s*N_s*Ntot*Ntotsave], matout[0:N_s*N_s*Ntot*Ntotsave],prod_norm_delta[0:Ntot*N_s*N_s], phiout[0:Ntot*N_s*N_s*Ntotsave],rhoout[0:Ntot*N_s*N_s*Ntotsave], lumout[0:N_s*N_s*Ntotsave],ECMout[0:N_s*N_s*Ntotsave]) create(lumn[0:N_s*N_s], ECMn[0:N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],f0[0:Ntot],s[0:Ntot],sum_s, psi[0:N_s*N_s], temp_phase[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], norm_grad_lum[0:N_s*N_s], J_x[0:Ntot*N_s*N_s], J_y[0:Ntot*N_s*N_s], hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s], degrates[0:Ntot*N_s*N_s], matgenrates[0:Ntot*N_s*N_s], viscosity[0:Ntot*N_s*N_s], b[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], pi_xx[0:Ntot*N_s*N_s], pi_xy[0:Ntot*N_s*N_s], pi_yy[0:Ntot*N_s*N_s], temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], temp_fft_real[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho[0:(2*zone_size-2)*(2*zone_size-2)], fft_react[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_xx[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_xy[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_yy[0:(2*zone_size-2)*(2*zone_size-2)], fft_vx[0:(2*zone_size-2)*(2*zone_size-2)], fft_vy[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vx[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vy[0:(2*zone_size-2)*(2*zone_size-2)])
    {

    fprintf(stderr,"Creating fft plans.\n \n");
    cufftPlan2d(&planf, 2*zone_size-2,2*zone_size-2, CUFFT_C2C);  // set the "plan" variable planf to be CUDA_FFT_complex_number
	cufftPlan2d(&planr, 2*zone_size-2,2*zone_size-2, CUFFT_C2C);


    fprintf(stderr,"Copy initial values to ECMn and lumn as well as phin and rhon.\n \n");
    copy_tables_ECM_or_lumen(ECMn, ECM);
    copy_tables_ECM_or_lumen(lumn, lum);
    for (int n=0; n<Ntot; n++)
    {
        copy_cell_tables_1_by_1(n, phin, phi);
        copy_cell_tables_1_by_1(n, rhon, rho);
    }

    
    it = 0;
    while ((it<Nsteps))  
    {   
        
        if (it==0)
        {
            fprintf(stderr,"I have started integrating ! \n");
        }

        if(it%nsaving == 0)
        {   
            fprintf(stderr,"I am saving in phiout, rhoout, lumout, ECMout  at time t = %d ! \n", it);
            saving_out_functions(it, phi, rho, ECM, lum, phiout, rhoout, lumout, ECMout); 
            // if also you want to saveout the maps of degeneration, generation rates and active gel parameters:
            //saving_turnover_rates_zones(it, rho, rhon, matout, degout);                 // use saving_turnover_rates_zones from useful_functions.cpp
        }

        if (it%ndisplay == 0)
        {
            fprintf(stderr,"Computing xcenters, ycenters. \n");
        }
        
        find_centres_cells(phi, xcenters, ycenters); //find the centre of mass of the phase field of each cell to then compute functions on restricted squares around these centerss

        if (it%ndisplay == 0)
        {
            fprintf(stderr,"Computing degrates, matgenrates, a_tan_1, a_ort_1, viscosity, b, maps. \n");
        }
        // create spatial maps for the different active gel parameters based on the cells, lumen and ECM phase fields
        create_turnover_rates_tensions(degrates, matgenrates, a_tan_1, a_ort_1, viscosity, b, phi, lum, ECM, xcenters, ycenters);
        
        // compute relevant functions for updating active gel densities, cell, lumen and ECM phase fields
        compute_hfunc_3d_tables_kernels_loop_n(hfunc_cell, phi, xcenters, ycenters);
        compute_all_hfunc_2d_tables_kernels(hfunc_lum, lum, hfunc_ECM, ECM);
        compute_normal_grad_phi(norm_grad_phi, phi);
        boundary_conditions_cells(norm_grad_phi);
        compute_prod_norm_delta(prod_norm_delta, norm_grad_phi, phi, xcenters, ycenters);

        compute_psi_kernels(psi, hfunc_cell);

        // this step computes the surface of each cell and update the value to the CPU, might be skipped to improve computational time
        sum_s_kernels(hfunc_cell, s, f0);

        if (it%ndisplay==0)
        {
        #pragma acc update host(s[0:Ntot])
        for (int n = 0; n<N_current; n++)
        {fprintf(stderr, "Surface of the cells at the step %09d ! It is : \n", it);
        fprintf(stderr, "s%d = %f ; \n", n+1, s[n]);
        }
        }

        // update all active gel densities in cortices (cf rho_functions.cpp)
        update_all_cortices(phi, rho, rhon, xcenters, ycenters, degrates, matgenrates, a_tan_1, a_ort_1, viscosity, b, norm_grad_phi, prod_norm_delta, planf, planr, temp_fft, temp_fft_real, fft_rho, fft_pi_xx, fft_pi_xy, fft_pi_yy, fft_vx, fft_vy, fft_react, fft_rho_vx, fft_rho_vy);

        // update lumen and ECM phase fields (cf cell_phase_functions.cpp)
        update_lumen_and_ECM(lum,lumn,ECM, ECMn, hfunc_lum, hfunc_ECM, psi);
        boundary_conditions_lumen_and_ECM(lumn,ECMn);
        
        reshaping_lumen_or_ECM(it,lum,lumn, 0.0);
        reshaping_lumen_or_ECM(it,ECM,ECMn, 1.0);
        
        // compute the normals to all cell phases (cf useful_functions.cpp)
        compute_normal_to_phi(phi, norm_grad_phi, J_x, J_y);
        boundary_conditions_cells(J_x);
        boundary_conditions_cells(J_y);
        
        // update lumen and ECM phase fields (cf cell_phase_functions.cpp)
        update_all_cells(it,  phi, phin, norm_grad_phi, rho, hfunc_cell, f0, psi, xcenters, ycenters, hfunc_lum, hfunc_ECM);
        boundary_conditions_cells(phin);
        
        reshaping_cells(it, phi, phin, norm_grad_phi, J_x, J_y, xcenters, ycenters);
        
        //copy rhon to rho before before moving to the next step
        for (int n=0; n<Ntot; n++)
        {
            copy_cell_tables_1_by_1(n, rho, rhon);
        }

        //copy ECMn to ECM and lumn to lum before moving to the next step
        copy_tables_ECM_or_lumen(ECM, ECMn);
        copy_tables_ECM_or_lumen(lum, lumn);
        
        it++;
    } //end while loop
    

    }
    #pragma acc exit data delete(degout[0:N_s*N_s*Ntot*Ntotsave], matout[0:N_s*N_s*Ntot*Ntotsave],prod_norm_delta[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phin[0:Ntot*N_s*N_s], rhon[0:Ntot*N_s*N_s], lumn[0:N_s*N_s], ECMn[0:N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],f0[0:Ntot],s[0:Ntot],sum_s, psi[0:N_s*N_s], temp_phase[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], norm_grad_lum[0:N_s*N_s], J_x[0:Ntot*N_s*N_s], J_y[0:Ntot*N_s*N_s], hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s], degrates[0:Ntot*N_s*N_s], matgenrates[0:Ntot*N_s*N_s], viscosity[0:Ntot*N_s*N_s], b[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], pi_xx[0:Ntot*N_s*N_s], pi_xy[0:Ntot*N_s*N_s], pi_yy[0:Ntot*N_s*N_s], temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], temp_fft_real[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho[0:(2*zone_size-2)*(2*zone_size-2)], fft_react[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_xx[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_xy[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_yy[0:(2*zone_size-2)*(2*zone_size-2)], fft_vx[0:(2*zone_size-2)*(2*zone_size-2)], fft_vy[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vx[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vy[0:(2*zone_size-2)*(2*zone_size-2)])
    
fprintf(stderr,"OUT OF GPU ! Let's save ! \n\n\n");

    for(it=0;it<(1+(int)((Nsteps-1)/nsaving));it++)
    {
    //fprintf(stderr,"Initial conditions every Nsaving saving ini_cell! \n");
    for(int n=0;n<N_current;n++){
    
    snprintf(filename, sizeof(filename), "ini_cell_%d.bin", n+1);
    std::ofstream cell_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    cell_next.write(reinterpret_cast<char const*>(&phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]), sizeof(&phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]));
    }}
    cell_next.close();

    //fprintf(stderr,"Initial conditions every Nsaving saving ini_rho! \n");
    snprintf(filename, sizeof(filename),"ini_rho_%d.bin", n+1);
    std::ofstream rho_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    rho_next.write(reinterpret_cast<char const*>(&rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]), sizeof(&rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]));
    }}
    rho_next.close();

    //fprintf(stderr,"Initial conditions every Nsaving saving cell ! \n");

    snprintf(filename, sizeof(filename),"data_int/phi_cell_%d_step_%d.dat", n+1, it*nsaving);
    ini_phi_out = fopen(filename, "w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_phi_out,"%.15f\n",phiout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
    fprintf(ini_phi_out,"\n");}
    fclose(ini_phi_out);
    
    //fprintf(stderr,"Initial conditions every Nsaving saving rho ! \n");

    snprintf(filename, sizeof(filename),"data_int/rho_cell_%d_step_%d.dat", n+1, it*nsaving);
    ini_rho_out = fopen(filename, "w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_rho_out,"%.15f\n",rhoout[it*N_s*N_s*Ntot+n*N_s*N_s+i*N_s+j]);}
    fprintf(ini_rho_out,"\n");}
    fclose(ini_rho_out);

    }   

    //fprintf(stderr,"Initial conditions every Nsaving saving ini lumen ! \n");

    sprintf(filename,"ini_lumen.bin");    
    std::ofstream lumen_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    lumen_next.write(reinterpret_cast<char const*>(&lumout[it*N_s*N_s + i*N_s+j]), sizeof(&lumout[it*N_s*N_s + i*N_s+j]));
    }}
    lumen_next.close();
    
    //fprintf(stderr,"Initial conditions every Nsaving saving ini ECM ! \n");

    sprintf(filename,"ini_ECM.bin");    
    std::ofstream ecm_next(filename, std::ios::binary);
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    ecm_next.write(reinterpret_cast<char const*>(&ECMout[it*N_s*N_s + i*N_s+j]), sizeof(&ECMout[it*N_s*N_s + i*N_s+j]));
    }}
    ecm_next.close();

    //fprintf(stderr,"Initial conditions every Nsaving saving lumen ! \n");

    sprintf(filename,"data_int/lumen_step_%d.dat",it*nsaving);
    ini_lumen_out = fopen(filename,"w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_lumen_out,"%.15f\n",lumout[it*N_s*N_s + i*N_s+j]);}
    fprintf(ini_lumen_out,"\n");}
    fclose(ini_lumen_out);

    //fprintf(stderr,"Initial conditions every Nsaving saving ECm ! \n");

    sprintf(filename,"data_int/ECM_step_%d.dat",it*nsaving);
    ini_ECM_out = fopen(filename,"w");
    for(int i=0;i<N_s;i++){for(int j=0;j<N_s;j++){
    fprintf(ini_ECM_out,"%.15f\n",ECMout[it*N_s*N_s + i*N_s+j]);}
    fprintf(ini_ECM_out,"\n");}
    fclose(ini_ECM_out);     

    }
    

    fprintf(stderr,"Let's free the variables ! \n");


    free(phi);
    free(phin);
    free(rho);
    free(rhon);
    free(J_x);
    free(J_y);
    free(prod_norm_delta);
    free(norm_grad_phi);
    free(hfunc_cell);
    free(lum);
    free(lumn);
    free(ECM);
    free(ECMn);
    free(psi);
    free(hfunc_lum);
    free(hfunc_ECM);
    free(temp_phase);

    free(phiout);
    free(rhoout);
    free(lumout);
    free(ECMout);
    free(degout);
    free(matout);

    free(f0);
    free(s);

    free(xcenters);
    free(ycenters);


    free(norm_grad_lum);
    free(degrates);
    free(matgenrates);
    free(viscosity);
    free(b);

    free(a_tan_1);
    free(a_ort_1);
    free(pi_xx);
    free(pi_xy);
    free(pi_yy);
    free(temp_fft);
    free(temp_fft_real);

    free(fft_rho);
    free(fft_react);
    free(fft_pi_xx);
    free(fft_pi_xy);
    free(fft_pi_yy);
    free(fft_vx);
    free(fft_vy);
    free(fft_rho_vx);
    free(fft_rho_vy);
    
    fprintf(stderr,"How much time was spent ? \n");

    /* Option : If you want to measure the time */
    timef = time(NULL);
    procf = clock();
    fprintf(stderr, "run time %d [sec] (end %d - start %d)\n", (int)(timef-timei), timef, timei);
    fprintf(stderr, "proc time %g [sec]\n", (float)(procf-proci)/(float)CLOCKS_PER_SEC);
    /* end of the option */    

    return(0);


}
