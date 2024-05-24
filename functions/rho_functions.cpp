#include <stdio.h>
#include <math.h>

#include <stdlib.h>
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <openacc.h> // used for OpenACC

#include "constants.h"
#include "declare_tables.h"
#include "useful_functions.h"

void compute_prod_norm_delta(float *prod_norm_delta, float *norm_grad_phi, float *phi, int *xcenters, int *ycenters)
{   
    
    #pragma acc kernels present(prod_norm_delta[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int n=0;n<Ntot;n++){
    #pragma acc loop independent
        for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size;i++){
            #pragma acc loop independent
            for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size;j++){

                prod_norm_delta[n*N_s*N_s+i*N_s+j] = norm_grad_phi[n*N_s*N_s+i*N_s+j]*exp(-(phi[n*N_s*N_s+i*N_s+j]-0.5)*(phi[n*N_s*N_s+i*N_s+j]-0.5)/2.0/sigma/sigma)/sqrt(2.0*M_PI)/sigma;

                }}
    }
    }
}

void compute_pi_xx(int n , float *pi_xx, float *rho, float *norm_grad_phi, float *phi, float *a_tan_1, float *a_ort_1, float *b, int *xcenters, int *ycenters)
{   ///table==pi_xx, table1==rho, table2==prod_norm_delta, table3==norm_grad_phi, table4== phi
    int ip;
    int im;

    #pragma acc kernels present(pi_xx[0:N_s*N_s], rho[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],b[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size;i++){
    #pragma acc loop independent
    for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size;j++)
    {   //careful here, the size of the zone should not reach the boundaries of the simulated grid because of the ip, im, jp, jm !!

        ip = i+1; im = i-1;

        pi_xx[n*N_s*N_s+i*N_s+j] = -a_tan_1[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]+ b[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j];

        if(norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.0)
        {
            pi_xx[n*N_s*N_s+i*N_s+j] += (a_tan_1[n*N_s*N_s+i*N_s+j]-a_ort_1[n*N_s*N_s+i*N_s+j])*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*(1.0/(4.0*dx*dx))*(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+j])*(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+j])/norm_grad_phi[n*N_s*N_s+i*N_s+j]/norm_grad_phi[n*N_s*N_s+i*N_s+j];
        }

        }
        }
    }
}

void compute_pi_yy(int n, float *pi_yy, float *rho, float *norm_grad_phi, float *phi, float *a_tan_1, float *a_ort_1, float *b, int *xcenters, int *ycenters)
{   ///table==pi_yy, table1==rho, table2==prod_norm_delta, table3==norm_grad_phi, table4== phi

    int jp;
    int jm;
    #pragma acc kernels present(pi_yy[0:N_s*N_s], rho[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot],b[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size;i++)
    {
    #pragma acc loop independent
    for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size;j++)
    {   //careful here, the size of the zone should not reach the boundaries of the simulated grid because of the ip, im, jp, jm !!

        jp = j+1;jm = j-1;

        pi_yy[n*N_s*N_s+i*N_s+j] = -a_tan_1[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]+ b[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j];

        if(norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.0)
        {
            pi_yy[n*N_s*N_s+i*N_s+j] += (a_tan_1[n*N_s*N_s+i*N_s+j]-a_ort_1[n*N_s*N_s+i*N_s+j])*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*(1.0/(4.0*dx*dx))*(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm])*(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm])/norm_grad_phi[n*N_s*N_s+i*N_s+j]/norm_grad_phi[n*N_s*N_s+i*N_s+j];
        }

        }
    }
    }
}


void compute_pi_xy(int n, float *pi_xy, float *rho, float *norm_grad_phi, float *phi, float *a_tan_1, float *a_ort_1, int *xcenters, int *ycenters)
{   ///table==pi_xy, table1==rho, table2==prod_norm_delta, table3==norm_grad_phi, table4== phi
    int ip;
    int im;
    int jp;
    int jm;
    #pragma acc kernels present(pi_yy[0:N_s*N_s], rho[0:Ntot*N_s*N_s], phi[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size;i++)
    {
    #pragma acc loop independent
    for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size;j++)
    {   //careful here, the size of the zone should not reach the boundaries of the simulated grid because of the ip, im, jp, jm !!

        ip = i+1; im = i-1;jp = j+1;jm = j-1;

        pi_xy[n*N_s*N_s+i*N_s+j] = 0.0;

        if(norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.0)
        {
            pi_xy[n*N_s*N_s+i*N_s+j] += (a_tan_1[n*N_s*N_s+i*N_s+j]-a_ort_1[n*N_s*N_s+i*N_s+j])*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*rho[n*N_s*N_s+i*N_s+j]*(1.0/(4.0*dx*dx))*(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+j])*(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm])/norm_grad_phi[n*N_s*N_s+i*N_s+j]/norm_grad_phi[n*N_s*N_s+i*N_s+j];
        }

    }
    }
    }
}



void randomize_rho(int n, float *table)
{ /// table==rho[n]
    
    for(int i=0;i<N_s;i++)
    {   
        for (int j=0;j<N_s;j++)
        {
            table[n*N_s*N_s+i*N_s+j] += 0.01*(0.5-rand()/(float)RAND_MAX);
        }
        }
}



void update_rho_velocity(int cell_number, cufftComplex *fft_vx, cufftComplex *fft_vy, cufftComplex *fft_pi_xx, cufftComplex *fft_pi_yy,cufftComplex *fft_pi_xy, float *viscosity, int *xcenters, int *ycenters)
{   
    int ir;
    int jr;
    float kx;
    float ky;
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;
    
    #pragma acc kernels present(fft_vx[0:(2*zone_size-2)*(2*zone_size-2)],fft_vy[0:(2*zone_size-2)*(2*zone_size-2)],fft_pi_xx[0:(2*zone_size-2)*(2*zone_size-2)], fft_pi_yy[0:(2*zone_size-2)*(2*zone_size-2)],fft_pi_xy[0:(2*zone_size-2)*(2*zone_size-2)], viscosity[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
        #pragma acc loop independent
        for(int j=0;j<2*zone_size-2;j++){

            i_gridcell = i+1+xcenters[cell_number]-zone_size;
            j_gridcell = j+1+ycenters[cell_number]-zone_size;

            
            fft_vx[i*(2*zone_size-2)+j].x = 0.0;
            fft_vx[i*(2*zone_size-2)+j].y = 0.0;

            fft_vy[i*(2*zone_size-2)+j].x = 0.0;
            fft_vy[i*(2*zone_size-2)+j].y = 0.0;

            if (i<(2*zone_size-2)/2){ir=i;}
            else {ir = i-(2*zone_size-2);}
            if (j<(2*zone_size-2)/2){jr=j;}
            else {jr = j-(2*zone_size-2);}

            kx = (2.0*M_PI/((2*zone_size-2)*dx))*ir;
            ky = (2.0*M_PI/((2*zone_size-2)*dx))*jr;

            //Numerator
            //V_Y
            //REAL PART
            fft_vy[i*(2*zone_size-2)+j].x += viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*ky*(kx*fft_pi_xx[i*(2*zone_size-2)+j].y + ky*fft_pi_xy[i*(2*zone_size-2)+j].y);
            fft_vy[i*(2*zone_size-2)+j].x -= (friction + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx+ky*ky/2.0))*(ky*fft_pi_yy[i*(2*zone_size-2)+j].y + kx*fft_pi_xy[i*(2*zone_size-2)+j].y);
            fft_vy[i*(2*zone_size-2)+j].x /= -(4.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*kx*ky*ky + friction*friction +3.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*friction*(kx*kx+ky*ky) + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx*kx*kx + ky*ky*ky*ky));

            //IMAGINARY PART
            fft_vy[i*(2*zone_size-2)+j].y -= viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*ky*(kx*fft_pi_xx[i*(2*zone_size-2)+j].x + ky*fft_pi_xy[i*(2*zone_size-2)+j].x); 
            fft_vy[i*(2*zone_size-2)+j].y += (friction + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx+ky*ky/2.0))*(ky*fft_pi_yy[i*(2*zone_size-2)+j].x + kx*fft_pi_xy[i*(2*zone_size-2)+j].x);
            fft_vy[i*(2*zone_size-2)+j].y /= -(4.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*kx*ky*ky + friction*friction +3.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*friction*(kx*kx+ky*ky) + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx*kx*kx + ky*ky*ky*ky));

            //V_X
            //REAL PART
            fft_vx[i*(2*zone_size-2)+j].x += kx*fft_pi_xx[i*(2*zone_size-2)+j].y + ky*fft_pi_xy[i*(2*zone_size-2)+j].y - viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*ky*fft_vy[i*(2*zone_size-2)+j].x;
            fft_vx[i*(2*zone_size-2)+j].x /= friction + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx + ky*ky/2.0);
            
            //IMAGINARY PART
            fft_vx[i*(2*zone_size-2)+j].y += - kx*fft_pi_xx[i*(2*zone_size-2)+j].x - ky*fft_pi_xy[i*(2*zone_size-2)+j].x - viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*kx*ky*fft_vy[i*(2*zone_size-2)+j].y;
            fft_vx[i*(2*zone_size-2)+j].y /= friction + 2.0*viscosity[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*(kx*kx + ky*ky/2.0);
            }
        }
        }
}


void create_deg_gen(int cell_number, cufftComplex *temp_fft, float *rho, float *prod_norm_delta, float *degrates, float *matgenrates, int *xcenters, int *ycenters)
{   
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;
    #pragma acc kernels present(temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], rho[0:Ntot*N_s*N_s], prod_norm_delta[0:Ntot*N_s*N_s], degrates[0:Ntot*N_s*N_s], matgenrates[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
    #pragma acc loop independent
    for(int j=0;j<2*zone_size-2;j++){
        i_gridcell = i+1+xcenters[cell_number]-zone_size;
        j_gridcell = j+1+ycenters[cell_number]-zone_size;
        temp_fft[i*(2*zone_size-2)+j].x = - degrates[cell_number*N_s*N_s+ i_gridcell*N_s + j_gridcell]*rho[cell_number*N_s*N_s+i_gridcell*N_s + j_gridcell] + matgenrates[cell_number*N_s*N_s + i_gridcell*N_s+j_gridcell]*prod_norm_delta[cell_number*N_s*N_s+i_gridcell*N_s+ j_gridcell];
        temp_fft[i*(2*zone_size-2)+j].y = 0.0;
        }}
    }
}


    
void update_fft_rho(cufftComplex *fft_rho, cufftComplex *fft_react, cufftComplex *fft_rho_vx, cufftComplex *fft_rho_vy)
{   
    // reaction part
    #pragma acc kernels present(fft_rho[0:(2*zone_size-2)*(2*zone_size-2)], fft_react[0:(2*zone_size-2)*(2*zone_size-2)])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
    #pragma acc loop independent
    for(int j=0;j<2*zone_size-2;j++){
    fft_rho[i*(2*zone_size-2)+j].x += dt*fft_react[i*(2*zone_size-2)+j].x;
    fft_rho[i*(2*zone_size-2)+j].y += dt*fft_react[i*(2*zone_size-2)+j].y;
    }}
    }
    
    int ir;
    int jr;
    // velocity with viscosity
    #pragma acc kernels present(fft_rho[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vx[0:(2*zone_size-2)*(2*zone_size-2)], fft_rho_vy[0:(2*zone_size-2)*(2*zone_size-2)])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
    #pragma acc loop independent
    for(int j=0;j<2*zone_size-2;j++){
        if (i<(2*zone_size-2)/2){ir=i;}
        else {ir = i-(2*zone_size-2);}
        if (j<(2*zone_size-2)/2){jr=j;}
        else {jr = j-(2*zone_size-2);}

            // real part
            // velocity with viscosity
            fft_rho[i*(2*zone_size-2)+j].x += dt*(2.0*M_PI/((2*zone_size-2)*dx))*(ir*fft_rho_vx[i*(2*zone_size-2)+j].y+jr*fft_rho_vy[i*(2*zone_size-2)+j].y);

            // imaginary part
            // velocity with viscosity
            fft_rho[i*(2*zone_size-2)+j].y -= dt*(2.0*M_PI/((2*zone_size-2)*dx))*(ir*fft_rho_vx[i*(2*zone_size-2)+j].x+jr*fft_rho_vy[i*(2*zone_size-2)+j].x);
        }}
    }
    
}

void copy_to_rhon(int n, cufftComplex *temp_fft, float *rhon, int *xcenters, int *ycenters)
{   
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;
    #pragma acc kernels present(temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], rhon[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
        #pragma acc loop independent
        for(int j=0;j<2*zone_size-2;j++){
            i_gridcell = i+1+xcenters[n]-zone_size;
            j_gridcell = j+1+ycenters[n]-zone_size;
            rhon[n*N_s*N_s+i_gridcell*N_s+j_gridcell] = temp_fft[i*(2*zone_size-2)+j].x/((2*zone_size-2)*(2*zone_size-2));}}
}

void load_fft_rho_v(int cell_number, cufftComplex *temp_fft, float *rho, cufftComplex *temp_fft_real, int *xcenters, int *ycenters)
{   
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;
    #pragma acc kernels present(temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], rho[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
    #pragma acc loop independent
    for(int j=0;j<2*zone_size-2;j++){
        i_gridcell = i+1+xcenters[cell_number]-zone_size;
        j_gridcell = j+1+ycenters[cell_number]-zone_size;
        temp_fft[i*(2*zone_size-2)+j].x = rho[cell_number*N_s*N_s+i_gridcell*N_s + j_gridcell]*temp_fft_real[i*(2*zone_size-2)+j].x/((2*zone_size-2)*(2*zone_size-2)); 
        temp_fft[i*(2*zone_size-2)+j].y = 0.0;
    }
    }
    }
}

void create_turnover_rates_tensions(float *degrates, float *matgenrates, float *a_tan_1, float *a_ort_1, float *viscosity, float *b, float *phi, float *lum, float *ECM, int *xcenters, int *ycenters)
{   
    int index = 0;
    float max = 0.0;
    int n1;
    int n2;
    float c1l, c1e,c1c2,c1c3;
    
    #pragma acc update host(xcenters[0:Ntot], ycenters[0:Ntot])

    for (int n=0; n<Ntot; n++)
    {
        if (n==0)
        {
            n1 = Ntot-1;
            n2 = n+1;
        }
        else if (n==Ntot-1)
        {
            n1 = n-1;
            n2 = 0;
        }
        else
        {
            n1 = n-1;
            n2 = n+1;
        }
    

    #pragma acc kernels present(phi[0:Ntot*N_s*N_s],degrates[0:Ntot*N_s*N_s], matgenrates[0:Ntot*N_s*N_s], a_tan_1[0:Ntot*N_s*N_s], a_ort_1[0:Ntot*N_s*N_s], lum[0:N_s*N_s], ECM[0:N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot], viscosity[0:Ntot*N_s*N_s], b[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size;i++){
    #pragma acc loop independent
    for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size;j++)
    { 
        //matgenrates[n*N_s*N_s+i*N_s+j] = matgenrate;

        max = 0.0;
        c1l = phi[n*N_s*N_s+ i*N_s+j]*lum[i*N_s+j];
        c1e = phi[n*N_s*N_s+ i*N_s+j]*ECM[i*N_s+j];
        
        if (c1l>c1e)
        {
            index = 0;
            max = c1l;
        }
        else
        {
            index = 1;
            max = c1e;
        }

        c1c2 = phi[n*N_s*N_s+ i*N_s+j]*phi[n1*N_s*N_s+i*N_s+j];
        
        if (c1c2>max)
        {
            index = 2;
            max = c1c2;
        }
        
        c1c3 = phi[n*N_s*N_s+ i*N_s+j]*phi[n2*N_s*N_s+i*N_s+j];
        
        if (c1c3>max)
        {
            index = 3;
            max = c1c3;
        }
        
        if (index == 2) //lateral
        {   
            degrates[n*N_s*N_s+i*N_s+j] = degrate_lateral;
            matgenrates[n*N_s*N_s+i*N_s+j] = matgenrate_lateral;
            a_ort_1[n*N_s*N_s+i*N_s+j] = a_ort_lateral;
            a_tan_1[n*N_s*N_s+i*N_s+j] = a_tan_lateral;
            viscosity[n*N_s*N_s+i*N_s+j] = viscosity_lat;
            b[n*N_s*N_s+i*N_s+j] = b_lat;

        }
        
        if  (index == 3) //lateral
        {
            degrates[n*N_s*N_s+i*N_s+j] = degrate_lateral;            
            matgenrates[n*N_s*N_s+i*N_s+j] = matgenrate_lateral;
            a_ort_1[n*N_s*N_s+i*N_s+j] = a_ort_lateral;
            a_tan_1[n*N_s*N_s+i*N_s+j] = a_tan_lateral;
            viscosity[n*N_s*N_s+i*N_s+j] = viscosity_lat;
            b[n*N_s*N_s+i*N_s+j] = b_lat;

        }
        if (index == 0) //apical
        {
            degrates[n*N_s*N_s+i*N_s+j] = degrate_apical;
            matgenrates[n*N_s*N_s+i*N_s+j] = matgenrate_apical;
            a_ort_1[n*N_s*N_s+i*N_s+j] = a_ort_apical;
            a_tan_1[n*N_s*N_s+i*N_s+j] = a_tan_apical;
            viscosity[n*N_s*N_s+i*N_s+j] = viscosity_ap;
            b[n*N_s*N_s+i*N_s+j] = b_ap;

        }

        if (index == 1) //basal
        {
            degrates[n*N_s*N_s+i*N_s+j] = degrate_basal;
            //matgenrates[n*N_s*N_s+i*N_s+j] = turnover_ratio*matgenrate;

            matgenrates[n*N_s*N_s+i*N_s+j] = matgenrate_basal;
            a_ort_1[n*N_s*N_s+i*N_s+j] = a_ort_basal;
            a_tan_1[n*N_s*N_s+i*N_s+j] = a_tan_basal;
            viscosity[n*N_s*N_s+i*N_s+j] = viscosity_bas;
            b[n*N_s*N_s+i*N_s+j] = b_bas;

        }
        

    }
    }
    }
    }
}



void update_all_cortices(float *phi, float *rho, float *rhon, int *xcenters, int *ycenters, float *degrates, float *matgenrates, float *a_tan_1, float *a_ort_1, float *viscosity, float *b, float *norm_grad_phi, float *prod_norm_delta, cufftHandle planf, cufftHandle planr, cufftComplex *temp_fft, cufftComplex *temp_fft_real, cufftComplex *fft_rho, cufftComplex *fft_pi_xx, cufftComplex *fft_pi_xy, cufftComplex *fft_pi_yy, cufftComplex *fft_vx, cufftComplex *fft_vy, cufftComplex *fft_react, cufftComplex *fft_rho_vx, cufftComplex *fft_rho_vy)
{   
    for (int nc=0; nc<Ntot; nc++)
        {   

            //Fourier transform the flux and rho
            load_temp_fft(nc, rho, temp_fft, xcenters, ycenters);

            compute_fft_forward(planf, temp_fft, fft_rho);

            
            create_deg_gen(nc, temp_fft, rho, prod_norm_delta, degrates, matgenrates, xcenters, ycenters);

            compute_fft_forward(planf, temp_fft, fft_react);     
           
            
           
            //Compute pi_xx, pi_xy, pi_yy for the cell
            
            compute_pi_xx(nc , pi_xx, rho, norm_grad_phi, phi, a_tan_1, a_ort_1, b, xcenters, ycenters);
            compute_pi_xy(nc , pi_xy, rho, norm_grad_phi, phi, a_tan_1, a_ort_1, xcenters, ycenters);
            compute_pi_yy(nc , pi_yy, rho, norm_grad_phi, phi, a_tan_1, a_ort_1, b, xcenters, ycenters);
            
            //Compute the ffts of pi_xx, pi_yy, pi_xy

            load_temp_fft(nc, pi_xx, temp_fft, xcenters, ycenters);
            compute_fft_forward(planf, temp_fft, fft_pi_xx);

            load_temp_fft(nc, pi_xy, temp_fft, xcenters, ycenters);
            compute_fft_forward(planf, temp_fft, fft_pi_xy);
            
            load_temp_fft(nc, pi_yy, temp_fft, xcenters, ycenters);
            compute_fft_forward(planf, temp_fft, fft_pi_yy);
            
            //compute fft_vx, fft_vy

            update_rho_velocity(nc, fft_vx, fft_vy, fft_pi_xx, fft_pi_yy, fft_pi_xy, viscosity, xcenters, ycenters);

            //compute vx then compute fft_rho_vx

            compute_fft_backward(planr, fft_vx, temp_fft_real);
            load_fft_rho_v(nc, temp_fft, rho, temp_fft_real, xcenters, ycenters);
            compute_fft_forward(planf, temp_fft, fft_rho_vx);

            //compute vy then compute fft_rho_vy

            compute_fft_backward(planr, fft_vy, temp_fft_real);
            load_fft_rho_v(nc, temp_fft, rho, temp_fft_real, xcenters, ycenters);
            compute_fft_forward(planf, temp_fft, fft_rho_vy);

            
            //update of rho drho/dt = -div(rho*v) + SOURCE TERM - SINK TERM

            update_fft_rho(fft_rho, fft_react, fft_rho_vx, fft_rho_vy);
            
            compute_fft_backward(planr, fft_rho, temp_fft);

            copy_to_rhon(nc, temp_fft, rhon, xcenters, ycenters);
        }
        
        boundary_conditions_cells(rhon);

}

