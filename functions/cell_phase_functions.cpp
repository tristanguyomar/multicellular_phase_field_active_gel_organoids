#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <openacc.h> // used for OpenACC

#include "constants.h"
#include "declare_tables.h"
#include "useful_functions.h"



#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

void print_mem(int it, char* name)
{   
    cout <<name<<"\n";
    cout << "Mem allocation at time = " << it << "\n";
    int tSize = 0, resident = 0, share = 0;
    ifstream buffer("/proc/self/statm");
    buffer >> tSize >> resident >> share;
    buffer.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    float rss = resident * page_size_kb;
    cout << "RSS - " << rss << " kB\n";

    float shared_mem = share * page_size_kb;
    cout << "Shared Memory - " << shared_mem << " kB\n";

    cout << "Private Memory - " << rss - shared_mem << "kB\n";
}



void update_rho_to_shape(float *rho, float *phi, float rho0, float tol)
{   
    
    #pragma acc kernels present(rho[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s])
    {
    for (int n = 0; n<Ntot; n++)
    {
    #pragma acc loop independent
    for (int i=0;i<N_s;i++)
    {
    #pragma acc loop independent
    for (int j=0;j<N_s;j++)
    {
    if(phi[n*N_s*N_s+i*N_s+j]!=0.0)
    {rho[n*N_s*N_s+i*N_s+j] = rho0*exp(-1.0/(4.0*phi[n*N_s*N_s+i*N_s+j]*phi[n*N_s*N_s+i*N_s+j]*(3.0-2.0*phi[n*N_s*N_s+i*N_s+j])*(1.0-phi[n*N_s*N_s+i*N_s+j]*phi[n*N_s*N_s+i*N_s+j]*(3.0-2.0*phi[n*N_s*N_s+i*N_s+j]))+tol)+1.0/(1.0+tol));
    }
    if(rho[n*N_s*N_s+i*N_s+j] < rho0*exp(-1.0/(2.0*tol)+1.0/(1.0+tol))) {rho[n*N_s*N_s+i*N_s+j] = 0.0;}
    }}
    }
    }
}


void update_all_cells_only_target_volume_and_adhesion(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *vol_diff , float *psi, int *xcenters, int *ycenters)
{/// table==phi, table1==phin, table2==norm_grad_phi, table3==rho, table4==hfunc_cell, vol_diff==f0

    //update phi
    int ip;
    int im;
    int jp;
    int jm;

    #pragma acc update host(xcenters[0:Ntot], ycenters[0:Ntot])

    for (int n=0; n<Ntot; n++)
    {

    #pragma acc kernels present(temp_phase[0:Ntot*N_s*N_s],psi[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0;i<N_s;i++)
    {   
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        // compute for each cell n only once : temp_phase[i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        temp_phase[n*N_s*N_s+i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        }
    }
    }

    #pragma acc kernels present(hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s], vol_diff[0:Ntot],phin[0:Ntot*N_s*N_s],norm_grad_phi[0:Ntot*N_s*N_s],rho[0:Ntot*N_s*N_s],phi[0:Ntot*N_s*N_s],temp_phase[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {
        #pragma acc loop independent
        for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++)
        {
            ip = i+1; 
            im = i-1; 
            jp = j+1; 
            jm = j-1;

            phin[n*N_s*N_s+i*N_s+j] = phi[n*N_s*N_s+i*N_s+j];

            // integration : interaction with cells, lumen and ECM for cell1
            phin[n*N_s*N_s+i*N_s+j]+= (1.0/tau)*phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j])*dt*(
            //volumic growth
            phi[n*N_s*N_s+i*N_s+j] -0.5 + vol_diff[n]
            
            //cell-cell exclusion
            -beta*temp_phase[n*N_s*N_s+i*N_s+j]
            //cell-cell adhesion (9 point stencil for laplacian)
            +eta*(-12.0*temp_phase[n*N_s*N_s+i*N_s+j]+2.0*(temp_phase[n*N_s*N_s+ip*N_s+j]+temp_phase[n*N_s*N_s+im*N_s+j]+temp_phase[n*N_s*N_s+i*N_s+jp]+temp_phase[n*N_s*N_s+i*N_s+jm])+temp_phase[n*N_s*N_s+ip*N_s+jp]+temp_phase[n*N_s*N_s+ip*N_s+jm]+temp_phase[n*N_s*N_s+im*N_s+jm]+temp_phase[n*N_s*N_s+im*N_s+jp])/(4.0*dx*dx)

            );
            
            
        }}
    }
    }
}

void update_all_cells(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *vol_diff , float *psi, int *xcenters, int *ycenters, float* hfunc_lum, float *hfunc_ECM)
{   /// table==phi, table1==phin, table2==norm_grad_phi, table3==rho, table4==hfunc_cell, vol_diff==f0

    //update phi
    int ip;
    int im;
    int jp;
    int jm;

    #pragma acc update host(xcenters[0:Ntot], ycenters[0:Ntot])

    for (int n=0; n<Ntot; n++)
    {
    

    #pragma acc kernels present(temp_phase[0:Ntot*N_s*N_s],psi[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0;i<N_s;i++)
    {   
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        // compute for each cell n only once : temp_phase[i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        temp_phase[n*N_s*N_s+i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        }
    }
    }



    #pragma acc kernels present(hfunc_lum[0:N_s*N_s], hfunc_ECM[0:N_s*N_s], vol_diff[0:Ntot],phin[0:Ntot*N_s*N_s],norm_grad_phi[0:Ntot*N_s*N_s],rho[0:Ntot*N_s*N_s],phi[0:Ntot*N_s*N_s],temp_phase[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {
        #pragma acc loop independent
        for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++)
        {
            ip = i+1; 
            im = i-1; 
            jp = j+1; 
            jm = j-1;

            phin[n*N_s*N_s+i*N_s+j] = phi[n*N_s*N_s+i*N_s+j];


            //delta function exp(-(phi[i*N_s+j]-0.5)*(phi[i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma
            
            
            //coupling with active gel
            if (norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.0)
            {
            phin[n*N_s*N_s+i*N_s+j]-= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[n*N_s*N_s+i*N_s+j]-0.5)*(phi[n*N_s*N_s+i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            (
                            (rho[n*N_s*N_s+ip*N_s+j]-rho[n*N_s*N_s+im*N_s+j])*(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+j]) +
                            (rho[n*N_s*N_s+i*N_s+jp]-rho[n*N_s*N_s+i*N_s+jm])*(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm])
                            )/norm_grad_phi[n*N_s*N_s+i*N_s+j];
            
            }
            
    
            // integration : interaction with cells, lumen and ECM for cell1
            phin[n*N_s*N_s+i*N_s+j]+= (1.0/tau)*phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j])*dt*(
            //volumic growth
            //phi[n*N_s*N_s+i*N_s+j] -0.5 + //no need for this term since we are reshaping the interface
            vol_diff[n]
            
            //cell-cell exclusion
            -beta*temp_phase[n*N_s*N_s+i*N_s+j]
            //cell-cell adhesion (9 point stencil for laplacian)
            +eta*(-12.0*temp_phase[n*N_s*N_s+i*N_s+j]+2.0*(temp_phase[n*N_s*N_s+ip*N_s+j]+temp_phase[n*N_s*N_s+im*N_s+j]+temp_phase[n*N_s*N_s+i*N_s+jp]+temp_phase[n*N_s*N_s+i*N_s+jm])+temp_phase[n*N_s*N_s+ip*N_s+jp]+temp_phase[n*N_s*N_s+ip*N_s+jm]+temp_phase[n*N_s*N_s+im*N_s+jm]+temp_phase[n*N_s*N_s+im*N_s+jp])/(4.0*dx*dx)
            
            //other part of the adhesion erm to prevent divergence from this term // no need for this term since we are implementing external surface tension through coupling with the active gel
            //+gamma_cell*(-12.0*hfunc_cell[n*N_s*N_s+i*N_s+j]+2.0*(hfunc_cell[n*N_s*N_s+ip*N_s+j]+hfunc_cell[n*N_s*N_s+im*N_s+j]+hfunc_cell[n*N_s*N_s+i*N_s+jp]+hfunc_cell[n*N_s*N_s+i*N_s+jm])+hfunc_cell[n*N_s*N_s+ip*N_s+jp]+hfunc_cell[n*N_s*N_s+ip*N_s+jm]+hfunc_cell[n*N_s*N_s+im*N_s+jm]+hfunc_cell[n*N_s*N_s+im*N_s+jp])/(4.0*dx*dx)
            //cell-lumen exclusion
            -beta_lum_cells*hfunc_lum[i*N_s+j]
            //cell-lumen adhesion
            +eta_lum_cells*(-12.0*hfunc_lum[i*N_s+j]+2.0*(hfunc_lum[ip*N_s+j]+hfunc_lum[im*N_s+j]+hfunc_lum[i*N_s+jp]+hfunc_lum[i*N_s+jm])+hfunc_lum[ip*N_s+jp]+hfunc_lum[ip*N_s+jm]+hfunc_lum[im*N_s+jm]+hfunc_lum[im*N_s+jp])/(4.0*dx*dx)
            //cell-ECM exclusion
            -beta_ECM_cells*hfunc_ECM[i*N_s+j]
            //cell-ECM adhesion
            +eta_ECM_cells*(-12.0*hfunc_ECM[i*N_s+j]+2.0*(hfunc_ECM[ip*N_s+j]+hfunc_ECM[im*N_s+j]+hfunc_ECM[i*N_s+jp]+hfunc_ECM[i*N_s+jm])+hfunc_ECM[ip*N_s+jp]+hfunc_ECM[ip*N_s+jm]+hfunc_ECM[im*N_s+jm]+hfunc_ECM[im*N_s+jp])/(4.0*dx*dx)
            
            );
            
            
        }}
    }
    }
}



void update_cells(int it, float *phi, float *phin, float *norm_grad_phi, float *rho, float *hfunc_cell, float *vol_diff , float *psi, float *hfunc_lum, float *hfunc_ECM)
{   /// table==phi, table1==phin, table2==norm_grad_phi, table3==rho, table4==hfunc_cell, vol_diff==f0

    //update phi
    int ip;
    int im;
    int jp;
    int jm;
    

    for (int n=0; n<Ntot; n++)
    {
    
    #pragma acc kernels present(temp_phase[0:Ntot*N_s*N_s],psi[0:N_s*N_s], hfunc_cell[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0;i<N_s;i++)
    {   
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        // compute for each cell n only once : temp_phase[i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        temp_phase[n*N_s*N_s+i*N_s+j] = psi[i*N_s+j] - hfunc_cell[n*N_s*N_s+i*N_s+j];
        //temp_phase[i*N_s+j] = psi[i*N_s+j] - hfunc_cell[i*N_s+j];
        //temp_phase[N_s*N_s+i*N_s+j] = psi[i*N_s+j] - hfunc_cell[N_s*N_s+i*N_s+j];
        }
    }
    }

    #pragma acc kernels present(vol_diff[0:Ntot],phin[0:Ntot*N_s*N_s],norm_grad_phi[0:Ntot*N_s*N_s],rho[0:Ntot*N_s*N_s],phi[0:Ntot*N_s*N_s],hfunc_ECM[0:N_s*N_s],hfunc_lum[0:N_s*N_s],temp_phase[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {   
        #pragma acc loop independent
        for(int j=1;j<N_s-1;j++)
        {
            ip = i+1; 
            im = i-1; 
            jp = j+1; 
            jm = j-1;

            phin[i*N_s+j] = phi[i*N_s+j];
            phin[N_s*N_s+i*N_s+j] = phi[N_s*N_s+i*N_s+j];


            //delta function exp(-(phi[i*N_s+j]-0.5)*(phi[i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma
            
            //coupling with active gel
            if (norm_grad_phi[i*N_s+j]>0.0)
            {
            phin[i*N_s+j]-= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[i*N_s+j]-0.5)*(phi[i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            (
                            (rho[ip*N_s+j]-rho[im*N_s+j])*(phi[ip*N_s+j]-phi[im*N_s+j]) +
                            (rho[i*N_s+jp]-rho[i*N_s+jm])*(phi[i*N_s+jp]-phi[i*N_s+jm])
                            )/norm_grad_phi[i*N_s+j];
            
            phin[i*N_s+j]-= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[i*N_s+j]-0.5)*(phi[i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            rho[i*N_s+j]*
                            (
                            -12.0*phi[i*N_s+j]+
                            2.0*(phi[ip*N_s+j]+phi[im*N_s+j]+phi[i*N_s+jp]+phi[i*N_s+jm])+
                            phi[ip*N_s+jp]+phi[ip*N_s+jm]+phi[im*N_s+jm]+phi[im*N_s+jp]
                            )/norm_grad_phi[i*N_s+j];
            
            phin[i*N_s+j]+= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[i*N_s+j]-0.5)*(phi[i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            rho[i*N_s+j]*
                            (
                            (phi[ip*N_s+j]-phi[im*N_s+j])*(norm_grad_phi[ip*N_s+j]-norm_grad_phi[im*N_s+j]) +
                            (phi[i*N_s+jp]-phi[i*N_s+jm])*(norm_grad_phi[i*N_s+jp]-norm_grad_phi[i*N_s+jm])
                            )/norm_grad_phi[i*N_s+j]/norm_grad_phi[i*N_s+j];
            
            }

            if (norm_grad_phi[N_s*N_s+i*N_s+j]>0.0)
            {
            phin[N_s*N_s+i*N_s+j]-= coupling_PF_AG/tau*
                                        (dt/dx/dx/4.0)*exp(-(phi[N_s*N_s+i*N_s+j]-0.5)*(phi[N_s*N_s+i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                                        (
                                        (rho[N_s*N_s+ip*N_s+j]-rho[N_s*N_s+im*N_s+j])*(phi[N_s*N_s+ip*N_s+j]-phi[N_s*N_s+im*N_s+j]) +
                                        (rho[N_s*N_s+i*N_s+jp]-rho[N_s*N_s+i*N_s+jm])*(phi[N_s*N_s+i*N_s+jp]-phi[N_s*N_s+i*N_s+jm])
                                        )/norm_grad_phi[N_s*N_s+i*N_s+j];

            
            phin[N_s*N_s+i*N_s+j]-= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[N_s*N_s+i*N_s+j]-0.5)*(phi[N_s*N_s+i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            rho[N_s*N_s+i*N_s+j]*
                            (
                            -12.0*phi[N_s*N_s+i*N_s+j]+
                            2.0*(phi[N_s*N_s+ip*N_s+j]+phi[N_s*N_s+im*N_s+j]+phi[N_s*N_s+i*N_s+jp]+phi[N_s*N_s+i*N_s+jm])+
                            phi[N_s*N_s+ip*N_s+jp]+phi[N_s*N_s+ip*N_s+jm]+phi[N_s*N_s+im*N_s+jm]+phi[N_s*N_s+im*N_s+jp]
                            )/norm_grad_phi[N_s*N_s+i*N_s+j];
            
            phin[N_s*N_s+i*N_s+j]+= coupling_PF_AG/tau*
                            (dt/dx/dx/4.0)*exp(-(phi[N_s*N_s+i*N_s+j]-0.5)*(phi[N_s*N_s+i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma*
                            rho[N_s*N_s+i*N_s+j]*
                            (
                            (phi[N_s*N_s+ip*N_s+j]-phi[N_s*N_s+im*N_s+j])*(norm_grad_phi[N_s*N_s+ip*N_s+j]-norm_grad_phi[N_s*N_s+im*N_s+j]) +
                            (phi[N_s*N_s+i*N_s+jp]-phi[N_s*N_s+i*N_s+jm])*(norm_grad_phi[N_s*N_s+i*N_s+jp]-norm_grad_phi[N_s*N_s+i*N_s+jm])
                            )/norm_grad_phi[N_s*N_s+i*N_s+j]/norm_grad_phi[N_s*N_s+i*N_s+j];
                  
            }
            
            
            // integration : interaction with cells, lumen and ECM for cell1
            phin[i*N_s+j]+= (1.0/tau)*phi[i*N_s+j]*(1.0-phi[i*N_s+j])*dt*(
            //volumic growth
            phi[i*N_s+j] -0.5 + vol_diff[0]
            //cell-cell exclusion
            -beta*temp_phase[i*N_s+j]
            //cell-cell adhesion (9 point stencil for laplacian)
            +eta*(-12.0*temp_phase[i*N_s+j]+2.0*(temp_phase[ip*N_s+j]+temp_phase[im*N_s+j]+temp_phase[i*N_s+jp]+temp_phase[i*N_s+jm])+temp_phase[ip*N_s+jp]+temp_phase[ip*N_s+jm]+temp_phase[im*N_s+jm]+temp_phase[im*N_s+jp])/(4.0*dx*dx)
            
            //cell-lumen exclusion
            //-beta_lum_cells*hfunc_lum[i*N_s+j]
            //cell-lumen adhesion
            //+eta_lum_cells*(-12.0*hfunc_lum[i*N_s+j]+2.0*(hfunc_lum[ip*N_s+j]+hfunc_lum[im*N_s+j]+hfunc_lum[i*N_s+jp]+hfunc_lum[i*N_s+jm])+hfunc_lum[ip*N_s+jp]+hfunc_lum[ip*N_s+jm]+hfunc_lum[im*N_s+jm]+hfunc_lum[im*N_s+jp])/(4.0*dx*dx)
            //cell-ECM exclusion
            //-beta_ECM_cells*hfunc_ECM[i*N_s+j]
            //cell-ECM adhesion
            //+eta_ECM_cells*(-12.0*hfunc_ECM[i*N_s+j]+2.0*(hfunc_ECM[ip*N_s+j]+hfunc_ECM[im*N_s+j]+hfunc_ECM[i*N_s+jp]+hfunc_ECM[i*N_s+jm])+hfunc_ECM[ip*N_s+jp]+hfunc_ECM[ip*N_s+jm]+hfunc_ECM[im*N_s+jm]+hfunc_ECM[im*N_s+jp])/(4.0*dx*dx)
            
            );
            
            // integration : interaction with cells, lumen and ECM for cell2
            phin[N_s*N_s+i*N_s+j]+= (1.0/tau)*phi[N_s*N_s+i*N_s+j]*(1.0-phi[N_s*N_s+i*N_s+j])*dt*(
            //volumic growth
            phi[N_s*N_s+i*N_s+j] -0.5 + vol_diff[1]
            //cell-cell exclusion
            -beta*temp_phase[N_s*N_s+i*N_s+j]
            //cell-cell adhesion (9 point stencil for laplacian)
            +eta*(-12.0*temp_phase[N_s*N_s+i*N_s+j]+2.0*(temp_phase[N_s*N_s+ip*N_s+j]+temp_phase[N_s*N_s+im*N_s+j]+temp_phase[N_s*N_s+i*N_s+jp]+temp_phase[N_s*N_s+i*N_s+jm])+temp_phase[N_s*N_s+ip*N_s+jp]+temp_phase[N_s*N_s+ip*N_s+jm]+temp_phase[N_s*N_s+im*N_s+jm]+temp_phase[N_s*N_s+im*N_s+jp])/(4.0*dx*dx)
            
            //cell-lumen exclusion
            //-beta_lum_cells*hfunc_lum[i*N_s+j]
            //cell-lumen adhesion
            //+eta_lum_cells*(-12.0*hfunc_lum[i*N_s+j]+2.0*(hfunc_lum[ip*N_s+j]+hfunc_lum[im*N_s+j]+hfunc_lum[i*N_s+jp]+hfunc_lum[i*N_s+jm])+hfunc_lum[ip*N_s+jp]+hfunc_lum[ip*N_s+jm]+hfunc_lum[im*N_s+jm]+hfunc_lum[im*N_s+jp])/(4.0*dx*dx)
            //cell-ECM exclusion
            //-beta_ECM_cells*hfunc_ECM[i*N_s+j]
            //cell-ECM adhesion
            //+eta_ECM_cells*(-12.0*hfunc_ECM[i*N_s+j]+2.0*(hfunc_ECM[ip*N_s+j]+hfunc_ECM[im*N_s+j]+hfunc_ECM[i*N_s+jp]+hfunc_ECM[i*N_s+jm])+hfunc_ECM[ip*N_s+jp]+hfunc_ECM[ip*N_s+jm]+hfunc_ECM[im*N_s+jm]+hfunc_ECM[im*N_s+jp])/(4.0*dx*dx)
            
            );
        }}
    }
    }
}

void update_lumen_and_ECM(float *lum, float *lumn, float *ECM, float *ECMn, float *hfunc_lum, float *hfunc_ECM, float *psi)
{   


    float sum_ecm=0.0;
    #pragma acc kernels present(hfunc_ECM[0:N_s*N_s])
    {
    sum_s=0.0;
    #pragma acc loop reduction(+:sum_ecm)
    for(int i=0;i<N_s;i++)
        {
        for(int j=0;j<N_s;j++)
            {   
                sum_ecm += (hfunc_ECM[i*N_s+j])*dx*dx;
        }}
    }
    

    int ip;
    int im;
    int jp;
    int jm;

    #pragma acc kernels present(lum[0:N_s*N_s],lumn[0:N_s*N_s],ECM[0:N_s*N_s],ECMn[0:N_s*N_s],hfunc_lum[0:N_s*N_s],hfunc_ECM[0:N_s*N_s],psi[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {   
        #pragma acc loop independent
        for(int j=1;j<N_s-1;j++)
        {
            ip = i+1; 
            im = i-1; 
            jp = j+1; 
            jm = j-1;

            lumn[i*N_s+j] = lum[i*N_s+j];
            lumn[i*N_s+j]+= (1.0/tau_lum)*lum[i*N_s+j]*(1.0-lum[i*N_s+j])*dt*(
            //lumen pressure
            //lum[i*N_s+j] - 0.5 + 
            xi_lum
            //lumen-cell exclusion
            -beta_lum_cells*psi[i*N_s+j]
            //lumen-cell adhesion
            +eta_lum_cells*(-12.0*psi[i*N_s+j]+2.0*(psi[ip*N_s+j]+psi[im*N_s+j]+psi[i*N_s+jp]+psi[i*N_s+jm])+psi[ip*N_s+jp]+psi[ip*N_s+jm]+psi[im*N_s+jm]+psi[im*N_s+jp])/(4.0*dx*dx)
            //non diverging term for adhesion
            //+gamma_lum_cells*(-12.0*hfunc_lum[i*N_s+j]+2.0*(hfunc_lum[ip*N_s+j]+hfunc_lum[im*N_s+j]+hfunc_lum[i*N_s+jp]+hfunc_lum[i*N_s+jm])+hfunc_lum[ip*N_s+jp]+hfunc_lum[ip*N_s+jm]+hfunc_lum[im*N_s+jm]+hfunc_lum[im*N_s+jp])/(4.0*dx*dx)
            
            //lumen-ECM exclusion
            -beta_lum_ECM*hfunc_ECM[i*N_s+j]
            );

            ECMn[i*N_s+j] = ECM[i*N_s+j];
            ECMn[i*N_s+j]+= (1.0/tau_ECM)*ECM[i*N_s+j]*(1.0-ECM[i*N_s+j])*dt*(
            //ECM elasticity
            alpha_ECM*(N_s*N_s*dx*dx-sum_ecm)
            //ECM pressure
            //+ ECM[i*N_s+j] - 0.5 
            + xi_ECM
            //ECM-cell exclusion
            -beta_ECM_cells*psi[i*N_s+j]
            //ECM-cell adhesion
            +eta_ECM_cells*(-12.0*psi[i*N_s+j]+2.0*(psi[ip*N_s+j]+psi[im*N_s+j]+psi[i*N_s+jp]+psi[i*N_s+jm])+psi[ip*N_s+jp]+psi[ip*N_s+jm]+psi[im*N_s+jm]+psi[im*N_s+jp])/(4.0*dx*dx)
            //non diverging term for adhesion
            //+gamma_ECM_cells*(-12.0*hfunc_ECM[i*N_s+j]+2.0*(hfunc_ECM[ip*N_s+j]+hfunc_ECM[im*N_s+j]+hfunc_ECM[i*N_s+jp]+hfunc_ECM[i*N_s+jm])+hfunc_ECM[ip*N_s+jp]+hfunc_ECM[ip*N_s+jm]+hfunc_ECM[im*N_s+jm]+hfunc_ECM[im*N_s+jp])/(4.0*dx*dx)
            //ECM-lumen exclusion
            -beta_lum_ECM*hfunc_lum[i*N_s+j]
            );

        }
    }
    }
}


void reshaping_cells(int it, float *phase, float *phasen, float *norm_grad_phi, float *J_x, float *J_y, int *xcenters, int *ycenters)
{   
    int ip;
    int im;
    int jp;
    int jm;

    int count_ri;

    int ri;
    float rdev;
    #pragma acc update host(xcenters[0:Ntot], ycenters[0:Ntot])
    // fprintf(stderr, "Starting reshaping at it = %d \n", it);

    for (int n=0; n<Ntot; n++)
    {
    if (it%ndisplay==0)
    {
    fprintf(stderr, "n = %d \n \n", n);

    fprintf(stderr, "xcenters[%d]-%d = %d  \n", n, zone_size, xcenters[n]-zone_size);
    fprintf(stderr, "xcenters[%d]+%d = %d  \n", n, zone_size, xcenters[n]+zone_size);
    fprintf(stderr, "ycenters[%d]-%d = %d  \n", n, zone_size, ycenters[n]-zone_size);
    fprintf(stderr, "ycenters[%d]+%d = %d  \n", n, zone_size, ycenters[n]+zone_size);
    }
    ri = 0;
    rdev = 2.0*rtol*2.0*M_PI*R*wi*rdt;
    count_ri = 0;
    while (rdev>rtol*2.0*M_PI*R*wi*rdt)
    {

    //fprintf(stderr, "Copying phasen in phase.\n");

    #pragma acc kernels present(phase[0:Ntot*N_s*N_s], phasen[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {

    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {
        #pragma acc loop independent
        for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++)
        {
            phase[n*N_s*N_s+i*N_s+j] = phasen[n*N_s*N_s+i*N_s+j];
        }
    }
    }
    

    //fprintf(stderr, "Updating phasen.\n");

    // J_x == n_x, J_y == n_y stands for the normal vector components
    #pragma acc kernels present(phase[0:Ntot*N_s*N_s],phasen[0:Ntot*N_s*N_s],J_x[0:Ntot*N_s*N_s],J_y[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {
        #pragma acc loop independent
        for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++)
    {
    ip = i+1;
    im = i-1;
    jp = j+1;
    jm = j-1;

    phasen[n*N_s*N_s+i*N_s+j] += 0.5*(J_x[n*N_s*N_s+ip*N_s+j]+J_x[n*N_s*N_s+i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phase[n*N_s*N_s+ip*N_s+j]-phase[n*N_s*N_s+i*N_s+j])*0.5*(J_x[n*N_s*N_s+ip*N_s+j]+J_x[n*N_s*N_s+i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phase[n*N_s*N_s+ip*N_s+jp]+phase[n*N_s*N_s+i*N_s+jp]-phase[n*N_s*N_s+ip*N_s+jm]-phase[n*N_s*N_s+i*N_s+jm])*0.5*(J_y[n*N_s*N_s+ip*N_s+j]+J_y[n*N_s*N_s+i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phase[n*N_s*N_s+ip*N_s+j]*(1.0-phase[n*N_s*N_s+ip*N_s+j])+phase[n*N_s*N_s+i*N_s+j]*(1.0-phase[n*N_s*N_s+i*N_s+j]))
        );


    phasen[n*N_s*N_s+i*N_s+j] += 0.5*(J_y[n*N_s*N_s+i*N_s+jp]+J_y[n*N_s*N_s+i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phase[n*N_s*N_s+i*N_s+jp]-phase[n*N_s*N_s+i*N_s+j])*0.5*(J_y[n*N_s*N_s+i*N_s+jp]+J_y[n*N_s*N_s+i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phase[n*N_s*N_s+ip*N_s+jp]+phase[n*N_s*N_s+ip*N_s+j]-phase[n*N_s*N_s+im*N_s+jp]-phase[n*N_s*N_s+im*N_s+j])*0.5*(J_x[n*N_s*N_s+i*N_s+jp]+J_x[n*N_s*N_s+i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phase[n*N_s*N_s+i*N_s+jp]*(1.0-phase[n*N_s*N_s+i*N_s+jp])+phase[n*N_s*N_s+i*N_s+j]*(1.0-phase[n*N_s*N_s+i*N_s+j]))
        );


    phasen[n*N_s*N_s+i*N_s+j] -= 0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+im*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phase[n*N_s*N_s+i*N_s+j]-phase[n*N_s*N_s+im*N_s+j])*0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+im*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phase[n*N_s*N_s+i*N_s+jp]+phase[n*N_s*N_s+im*N_s+jp]-phase[n*N_s*N_s+i*N_s+jm]-phase[n*N_s*N_s+im*N_s+jm])*0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+im*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phase[n*N_s*N_s+i*N_s+j]*(1.0-phase[n*N_s*N_s+i*N_s+j])+phase[n*N_s*N_s+im*N_s+j]*(1.0-phase[n*N_s*N_s+im*N_s+j]))
        );

    phasen[n*N_s*N_s+i*N_s+j] -= 0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+i*N_s+jm])*(
        wi*wi*(rdt/dx/dx)
        *(phase[n*N_s*N_s+i*N_s+j]-phase[n*N_s*N_s+i*N_s+jm])*0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+i*N_s+jm])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phase[n*N_s*N_s+ip*N_s+j]+phase[n*N_s*N_s+ip*N_s+jm]-phase[n*N_s*N_s+im*N_s+j]-phase[n*N_s*N_s+im*N_s+jm])*0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+i*N_s+jm])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phase[n*N_s*N_s+i*N_s+j]*(1.0-phase[n*N_s*N_s+i*N_s+j])+phase[n*N_s*N_s+i*N_s+jm]*(1.0-phase[n*N_s*N_s+i*N_s+jm]))
        );
    }}
    }

    // fprintf(stderr, "Computing Sum_s. \n");

    //relaxation check
    rdev = 0.0;
    #pragma acc kernels present(phasen[0:Ntot*N_s*N_s], phase[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop reduction(+:rdev)
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {for(int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++){
        rdev += (fabs(phasen[n*N_s*N_s+i*N_s+j]-phase[n*N_s*N_s+i*N_s+j]))*dx*dx/N_current;
    }}
    }


    if(ri > rimax){
    fprintf(stderr,"Relaxation failure! at i=%d ri=%d \n", it, ri);
    break;
    }
    
    // fprintf(stderr, "Before ri++. \n");

    ri++;
    count_ri +=1;
    }
    if (it%ndisplay==0)
    {
    fprintf(stderr, "count_ri = %d \n \n", count_ri);
    }
    }
}



void reshaping_cells_all_grid(int it, float *phi, float *phin, float *J_x, float *J_y)
{   
    int ip;
    int im;
    int jp;
    int jm;

    int count_ri;

    int ri;
    float rdev;
    // fprintf(stderr, "Starting reshaping at it = %d \n", it);

    for (int n=0; n<Ntot; n++)
    {
    fprintf(stderr, "n = %d \n \n", n);

    ri = 0;
    rdev = 2.0*rtol*2.0*M_PI*R*wi*rdt;
    count_ri = 0;
    while (rdev>rtol*2.0*M_PI*R*wi*rdt)
    {

    fprintf(stderr, "Copying phin in phi.\n");

    #pragma acc kernels present(phi[0:Ntot*N_s*N_s], phin[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0;i<N_s; i++)
    {
        #pragma acc loop independent
        for (int j=0;j<N_s; j++)
        {
            phi[n*N_s*N_s+i*N_s+j] = phin[n*N_s*N_s+i*N_s+j];
        }
    }
    }
    

    fprintf(stderr, "Updating phin.\n");

    // J_x == n_x, J_y == n_y stands for the normal vector components
    #pragma acc kernels present(phi[0:Ntot*N_s*N_s],phin[0:Ntot*N_s*N_s],J_x[0:Ntot*N_s*N_s],J_y[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1; i++)
    {
        #pragma acc loop independent
        for (int j=1;j<N_s-1; j++)
    {
    ip = i+1;
    im = i-1;
    jp = j+1;
    jm = j-1;

    phin[n*N_s*N_s+i*N_s+j] += 0.5*(J_x[n*N_s*N_s+ip*N_s+j]+J_x[n*N_s*N_s+i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+i*N_s+j])*0.5*(J_x[n*N_s*N_s+ip*N_s+j]+J_x[n*N_s*N_s+i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phi[n*N_s*N_s+ip*N_s+jp]+phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+ip*N_s+jm]-phi[n*N_s*N_s+i*N_s+jm])*0.5*(J_y[n*N_s*N_s+ip*N_s+j]+J_y[n*N_s*N_s+i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phi[n*N_s*N_s+ip*N_s+j]*(1.0-phi[n*N_s*N_s+ip*N_s+j])+phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j]))
        );


    phin[n*N_s*N_s+i*N_s+j] += 0.5*(J_y[n*N_s*N_s+i*N_s+jp]+J_y[n*N_s*N_s+i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+j])*0.5*(J_y[n*N_s*N_s+i*N_s+jp]+J_y[n*N_s*N_s+i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phi[n*N_s*N_s+ip*N_s+jp]+phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+jp]-phi[n*N_s*N_s+im*N_s+j])*0.5*(J_x[n*N_s*N_s+i*N_s+jp]+J_x[n*N_s*N_s+i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phi[n*N_s*N_s+i*N_s+jp]*(1.0-phi[n*N_s*N_s+i*N_s+jp])+phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j]))
        );


    phin[n*N_s*N_s+i*N_s+j] -= 0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+im*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(phi[n*N_s*N_s+i*N_s+j]-phi[n*N_s*N_s+im*N_s+j])*0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+im*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phi[n*N_s*N_s+i*N_s+jp]+phi[n*N_s*N_s+im*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm]-phi[n*N_s*N_s+im*N_s+jm])*0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+im*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j])+phi[n*N_s*N_s+im*N_s+j]*(1.0-phi[n*N_s*N_s+im*N_s+j]))
        );

    phin[n*N_s*N_s+i*N_s+j] -= 0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+i*N_s+jm])*(
        wi*wi*(rdt/dx/dx)
        *(phi[n*N_s*N_s+i*N_s+j]-phi[n*N_s*N_s+i*N_s+jm])*0.5*(J_y[n*N_s*N_s+i*N_s+j]+J_y[n*N_s*N_s+i*N_s+jm])
        +wi*wi*(rdt/dx/dx)
        *0.25*(phi[n*N_s*N_s+ip*N_s+j]+phi[n*N_s*N_s+ip*N_s+jm]-phi[n*N_s*N_s+im*N_s+j]-phi[n*N_s*N_s+im*N_s+jm])*0.5*(J_x[n*N_s*N_s+i*N_s+j]+J_x[n*N_s*N_s+i*N_s+jm])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(phi[n*N_s*N_s+i*N_s+j]*(1.0-phi[n*N_s*N_s+i*N_s+j])+phi[n*N_s*N_s+i*N_s+jm]*(1.0-phi[n*N_s*N_s+i*N_s+jm]))
        );
    }}
    }

    // fprintf(stderr, "Computing Sum_s. \n");

    //relaxation check
    rdev = 0.0;
    #pragma acc kernels present(phin[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s])
    {
    #pragma acc loop reduction(+:rdev)
    for (int i=0;i<N_s; i++)
    {for(int j=0;j<N_s; j++){
        rdev += (fabs(phin[n*N_s*N_s+i*N_s+j]-phi[n*N_s*N_s+i*N_s+j]))*dx*dx/N_current;
    }}
    }
    fprintf(stderr, "For cell number = %d at t_reshape = %d, rdev/r_devtarget = %f \n \n", n, count_ri, rdev/(rtol*2.0*M_PI*R*wi*rdt));


    if(ri > rimax){
    fprintf(stderr,"Relaxation failure! at i=%d ri=%d \n", it, ri);
    break;
    }
    
    // fprintf(stderr, "Before ri++. \n");

    ri++;
    count_ri +=1;
    }
    fprintf(stderr, "count_ri = %d \n \n", count_ri);
    }
}


void reshaping_lumen_or_ECM(int it, float *lum, float *lumn, float side_value)
{  


    int ip;
    int im;
    int jp;
    int jm;

    int ri = 0;
    float rdev;
    rdev = rdev_ini*rtol*rdt;

    #pragma acc kernels present(lum[0:N_s*N_s],norm_grad_lum[0:N_s*N_s])
    {   
        #pragma acc loop independent
        for(int i=1;i<N_s-1;i++)
        {
            #pragma acc loop independent
            for(int j=1;j<N_s-1;j++)
            {
                ip = i+1;
                im = i-1;
                jp = j+1;
                jm = j-1;
                norm_grad_lum[i*N_s+j] = sqrt( (1.0/dx/dx/4.0)* (
                        (lum[ip*N_s+j]-lum[im*N_s+j])*(lum[ip*N_s+j]-lum[im*N_s+j])+(lum[i*N_s+jp]-lum[i*N_s+jm])*(lum[i*N_s+jp]-lum[i*N_s+jm])
                        ));
            }
        }
    }

    boundary_conditions_periodic(norm_grad_lum);

    //compute normal vector : grad(phi)/norm_grad(phi) BEWARE : using J_x and J_y for memory commodities
    #pragma acc kernels present(norm_grad_lum[0:N_s*N_s], J_x[0:N_s*N_s],J_y[0:N_s*N_s],lum[0:N_s*N_s], lumn[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    #pragma acc loop independent
    for(int j=1;j<N_s-1;j++)
    {
    ip = i+1;
    im = i-1;
    jp = j+1;
    jm = j-1;

    if (norm_grad_lum[i*N_s+j]!=0.0)
    {J_x[i*N_s+j] = (1.0/(2.0*dx))*(lumn[ip*N_s+j]-lumn[im*N_s+j])/norm_grad_lum[i*N_s+j];}
    else {J_x[i*N_s+j]=0.0;}

    if (norm_grad_lum[i*N_s+j]!=0.0)
    {J_y[i*N_s+j] = (1.0/(2.0*dx))*(lumn[i*N_s+jp]-lumn[i*N_s+jm])/norm_grad_lum[i*N_s+j];}
    else {J_y[i*N_s+j]=0.0;}
    }}
    }
    

    //periodic boundary conditions
    
    boundary_conditions_periodic(J_x);
    boundary_conditions_periodic(J_y);


    while (rdev>rdev_threshold*rtol*rdt)
    {

    #pragma acc kernels present(lum[0:N_s*N_s], lumn[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0;i<N_s;i++){
    #pragma acc loop independent
    for(int j=0;j<N_s;j++){
        lum[i*N_s+j] = lumn[i*N_s+j];
    }}
    }
    
    // J_x == n_x, J_y == n_y stands for the normal vector components

    #pragma acc kernels present(lum[0:N_s*N_s],lumn[0:N_s*N_s],J_x[0:N_s*N_s],J_y[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    #pragma acc loop independent
    for(int j=1;j<N_s-1;j++)
    {
    ip = i+1;
    im = i-1;
    jp = j+1;
    jm = j-1;

    lumn[i*N_s+j] += 0.5*(J_x[ip*N_s+j]+J_x[i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(lum[ip*N_s+j]-lum[i*N_s+j])*0.5*(J_x[ip*N_s+j]+J_x[i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(lum[ip*N_s+jp]+lum[i*N_s+jp]-lum[ip*N_s+jm]-lum[i*N_s+jm])*0.5*(J_y[ip*N_s+j]+J_y[i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(lum[ip*N_s+j]*(1.0-lum[ip*N_s+j])+lum[i*N_s+j]*(1.0-lum[i*N_s+j]))
        );


    lumn[i*N_s+j] += 0.5*(J_y[i*N_s+jp]+J_y[i*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(lum[i*N_s+jp]-lum[i*N_s+j])*0.5*(J_y[i*N_s+jp]+J_y[i*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(lum[ip*N_s+jp]+lum[ip*N_s+j]-lum[im*N_s+jp]-lum[im*N_s+j])*0.5*(J_x[i*N_s+jp]+J_x[i*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(lum[i*N_s+jp]*(1.0-lum[i*N_s+jp])+lum[i*N_s+j]*(1.0-lum[i*N_s+j]))
        );


    lumn[i*N_s+j] -= 0.5*(J_x[i*N_s+j]+J_x[im*N_s+j])*(
        wi*wi*(rdt/dx/dx)
        *(lum[i*N_s+j]-lum[im*N_s+j])*0.5*(J_x[i*N_s+j]+J_x[im*N_s+j])
        +wi*wi*(rdt/dx/dx)
        *0.25*(lum[i*N_s+jp]+lum[im*N_s+jp]-lum[i*N_s+jm]-lum[im*N_s+jm])*0.5*(J_y[i*N_s+j]+J_y[im*N_s+j])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(lum[i*N_s+j]*(1.0-lum[i*N_s+j])+lum[im*N_s+j]*(1.0-lum[im*N_s+j]))
        );

    lumn[i*N_s+j] -= 0.5*(J_y[i*N_s+j]+J_y[i*N_s+jm])*(
        wi*wi*(rdt/dx/dx)
        *(lum[i*N_s+j]-lum[i*N_s+jm])*0.5*(J_y[i*N_s+j]+J_y[i*N_s+jm])
        +wi*wi*(rdt/dx/dx)
        *0.25*(lum[ip*N_s+j]+lum[ip*N_s+jm]-lum[im*N_s+j]-lum[im*N_s+jm])*0.5*(J_x[i*N_s+j]+J_x[i*N_s+jm])
        - rdt*wi*sqrt(2.0)/dx
        *0.5*(lum[i*N_s+j]*(1.0-lum[i*N_s+j])+lum[i*N_s+jm]*(1.0-lum[i*N_s+jm]))
        );
    }}
    }

    //boundary conditions 

    boundary_conditions_dirichlet(lumn, side_value);

    //relaxation check
    
    rdev = 0.0;
    #pragma acc kernels present(lum[0:N_s*N_s], lumn[0:N_s*N_s])
    {
    #pragma acc loop reduction(+:rdev)
    for (int i=0;i<N_s;i++)
    {for(int j=0;j<N_s;j++)
    {rdev += fabs(lumn[i*N_s+j]-lum[i*N_s+j])*dx*dx;}} 
    }
    //display 
    // if(ri%riprint==0 && it%ndisplay==0)
    if(ri > threshold_reshaping_steps && ri%riprint==0)
    {   
        fprintf(stderr,"Value to match for lumen relaxation is: rdev_theory = %lf . \n", rdev_threshold*rtol*rdt);
        fprintf(stderr,"Relaxation test! rdev=%lf ri=%d \n\n", rdev, ri);}
    if(ri > rimax){
    fprintf(stderr,"Relaxation failure! at i=%d ri=%d \n", it, ri);
    break;
    }
    

    ri++;
    }//end of while = end of reshaping loop
    
}
