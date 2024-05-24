#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "declare_tables.h"
#include "declare_fft_tables.h"
#include <cufft.h> // used for CUDA FFT
#include <cublas.h>
#include <cmath>


int check_nan(float *table)
{   int c1 = 0;
    int c2 = 0;
    for (int i =0; i<N_s;i++)
    {for (int j=0;j<N_s;j++)
        {
            if (isnan(table[i*N_s+j])){
                c1+=1;}
            if (isnan(table[N_s*N_s+i*N_s+j])){
                c2+=1;}

        }
    }
    return(c1+c2);
}

int check_initial_condition(float *table1, float *table2)
{   
    int c = 0;
    for(int i = 0; i<N_s; i++)
    {
        for(int j=0; j<N_s; j++)
        {
            if (table1[i*N_s+j] != table2[i*N_s+j])
            {
                c+=1; 
            }
        }
    }
    return(c);
}

void saving_turnover_rates_zones(int it, float *matgenrates, float *degrates, float *matout, float *degout)
{   

    #pragma acc kernels present(degrates[0:Ntot*N_s*N_s], matgenrates[0:Ntot*N_s*N_s], degout[0:Ntot*N_s*N_s*Ntotsave], matout[0:Ntot*N_s*N_s*Ntotsave])
    {   
        #pragma acc loop independent
        for (int n=0;n<N_ini;n++)
        {
        #pragma acc loop independent
        for(int i=0;i<N_s;i++)
        {
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        matout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n + i*N_s+j] = matgenrates[N_s*N_s*n + i*N_s+j];
        degout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n +i*N_s+j] = degrates[N_s*N_s*n + i*N_s+j];
        }}
        }
    }

}


void saving_out_functions(int it, float *phi, float *rho, float *ECM, float *lum, float *phiout, float *rhoout, float *lumout, float *ECMout)
{   

    #pragma acc kernels present(phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phiout[0:Ntot*N_s*N_s*Ntotsave],rhoout[0:Ntot*N_s*N_s*Ntotsave])
    {   
        #pragma acc loop independent
        for (int n=0;n<N_ini;n++)
        {
        #pragma acc loop independent
        for(int i=0;i<N_s;i++)
        {
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        phiout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n + i*N_s+j] = phi[N_s*N_s*n + i*N_s+j];
        rhoout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n +i*N_s+j] = rho[N_s*N_s*n + i*N_s+j];
        }}
        }
    }


    #pragma acc kernels present(ECM[0:N_s*N_s], lum[0:N_s*N_s],ECMout[0:Ntotsave*N_s*N_s], lumout[0:Ntotsave*N_s*N_s])
    {
        #pragma acc loop independent
        for(int i=0;i<N_s;i++)
        {
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        lumout[(int)(it/nsaving)*N_s*N_s+i*N_s+j] = lum[i*N_s+j];
        ECMout[(int)(it/nsaving)*N_s*N_s+i*N_s+j] = ECM[i*N_s+j];
        }}
    } 
}


void saving_out_functions_with_normals(int it, float *phi, float *rho, float *ECM, float *lum, float *phiout, float *rhoout, float *lumout, float *ECMout, float *J_x, float *J_y, float *Jxout, float *Jyout)
{   

    #pragma acc kernels present(phi[0:Ntot*N_s*N_s], rho[0:Ntot*N_s*N_s], phiout[0:Ntot*N_s*N_s*Ntotsave],rhoout[0:Ntot*N_s*N_s*Ntotsave], Jxout[0:Ntot*N_s*N_s*Ntotsave], Jyout[0:Ntot*N_s*N_s*Ntotsave])
    {   
        #pragma acc loop independent
        for (int n=0;n<N_ini;n++)
        {
        #pragma acc loop independent
        for(int i=0;i<N_s;i++)
        {
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        phiout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n + i*N_s+j] = phi[N_s*N_s*n + i*N_s+j];
        rhoout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n +i*N_s+j] = rho[N_s*N_s*n + i*N_s+j];
        Jxout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n +i*N_s+j] = J_x[N_s*N_s*n + i*N_s+j];
        Jyout[(int)(it/nsaving)*N_s*N_s*Ntot+ N_s*N_s*n +i*N_s+j] = J_y[N_s*N_s*n + i*N_s+j];
        }}
        }
    }


    #pragma acc kernels present(ECM[0:N_s*N_s], lum[0:N_s*N_s],ECMout[0:Ntotsave*N_s*N_s], lumout[0:Ntotsave*N_s*N_s])
    {
        #pragma acc loop independent
        for(int i=0;i<N_s;i++)
        {
        #pragma acc loop independent
        for(int j=0;j<N_s;j++)
        {
        lumout[(int)(it/nsaving)*N_s*N_s+i*N_s+j] = lum[i*N_s+j];
        ECMout[(int)(it/nsaving)*N_s*N_s+i*N_s+j] = ECM[i*N_s+j];
        }}
    } 
}


void psi_to_zero_kernels(float *psi)
{
    #pragma acc kernels present(psi[0:N_s*N_s])
    {
    for (int i=0; i<N_s; i++)
    {   
        for (int j=0;j<N_s;j++)
        {   
            psi[i*N_s+j] = 0.0;
        }
    }
    }    
}

void psi_to_zero(float *psi)
{   
    #pragma acc data present(psi[0:N_s*N_s])
    #pragma acc parallel loop collapse(2) device_type(nvidia) vector_length(128)
    for (int i=0; i<N_s; i++)
    {   
        for (int j=0;j<N_s;j++)
        {   
            psi[i*N_s+j] = 0.0;
        }
    }
}


void sum_s_kernels(float *hfunc_cell, float *s, float *f0)
{

    float sum_s=0.0;
    #pragma acc kernels present(hfunc_cell[0:Ntot*N_s*N_s],s[0:Ntot],f0[0:Ntot])
    {
    #pragma acc loop independent
    for(int n=0;n<Ntot;n++)
    {
    sum_s=0.0;
    #pragma acc loop reduction(+:sum_s)
    for(int i=0;i<N_s;i++)
        {
        for(int j=0;j<N_s;j++)
            {   
                sum_s += (hfunc_cell[n*N_s*N_s+i*N_s+j])*dx*dx;
        }}

    s[n] = sum_s;
    f0[n] = alpha*(s0-s[n]);
    }
    }
}


void find_centres_cells(float *phi, int *xcenters, int *ycenters)
{   
    int x_barycentre = 0;
    int y_barycentre = 0;
    int count = 0;

    #pragma acc kernels present(phi[0:Ntot*N_s*N_s], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for (int n =0; n<Ntot; n++)
    {   
        x_barycentre = 0;
        y_barycentre = 0;
        count = 0;
        #pragma acc loop reduction(+:x_barycentre, y_barycentre, count)
        for (int i = 0; i<N_s; i++)
        {
            for (int j = 0; j<N_s; j++)
            {
                if (phi[n*N_s*N_s+i*N_s+j] > 0.5)
                {
                    x_barycentre += i;
                    y_barycentre += j;
                    count +=1;
                }
            } 
        }

        xcenters[n] = (int)(x_barycentre/count);
        ycenters[n] = (int)(y_barycentre/count);
        

    }
    }
}





void compute_hfunc_3d_tables_kernels_loop_n(float *table1, float *table2, int *xcenters, int *ycenters)
{   // table1 = hfunc_lum | hfunc_ECM, table2 = lum | ECM

    #pragma acc kernels present(table1[0:Ntot*N_s*N_s],table2[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int n=0;n<N_ini;n++)
    {
    #pragma acc loop independent
    for (int i=xcenters[n]-zone_size;i<xcenters[n]+zone_size; i++)
    {
        #pragma acc loop independent
        for (int j=ycenters[n]-zone_size;j<ycenters[n]+zone_size; j++)
        {
        table1[n*N_s*N_s+i*N_s+j] = table2[n*N_s*N_s+i*N_s+j]* table2[n*N_s*N_s+i*N_s+j]*(3.0 - 2.0* table2[n*N_s*N_s+i*N_s+j]);
        }
    }}
    }
}



void compute_hfunc_3d_tables_kernels(float *table1, float *table2)
{   // table1 = hfunc_cell , table2 = phi
    #pragma acc kernels present(table1[0:Ntot*N_s*N_s],table2[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0; i<N_s; i++){
        #pragma acc loop independent
        for (int j=0;j<N_s;j++){
        
        table1[i*N_s+j] = table2[i*N_s+j]* table2[i*N_s+j]*(3.0 - 2.0* table2[i*N_s+j]);
        table1[N_s*N_s+i*N_s+j] = table2[N_s*N_s+i*N_s+j]* table2[N_s*N_s+i*N_s+j]*(3.0 - 2.0* table2[N_s*N_s+i*N_s+j]);

    }}
    }
}


void compute_all_hfunc_2d_tables_kernels(float *table1, float *table2, float *table3, float *table4)
{   // table1 = hfunc_lum | hfunc_ECM, table2 = lum | ECM
    #pragma acc kernels present(table1[0:N_s*N_s],table2[0:N_s*N_s],table3[0:N_s*N_s],table4[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0; i<N_s; i++){
        #pragma acc loop independent
        for (int j=0;j<N_s;j++){
        table1[i*N_s+j] = table2[i*N_s+j]* table2[i*N_s+j]*(3.0 - 2.0* table2[i*N_s+j]);
        table3[i*N_s+j] = table4[i*N_s+j]* table4[i*N_s+j]*(3.0 - 2.0* table4[i*N_s+j]);
    }}
    }
    // cette fonction est deux fois plus rapide que deux appel à compute_hfunc_2d_tables_kernels
}



void compute_hfunc_2d_tables_kernels(float *table1, float *table2)
{   // table1 = hfunc_lum | hfunc_ECM, table2 = lum | ECM
    #pragma acc kernels present(table1[0:N_s*N_s],table2[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0; i<N_s; i++){
        #pragma acc loop independent
        for (int j=0;j<N_s;j++){
        table1[i*N_s+j] = table2[i*N_s+j]* table2[i*N_s+j]*(3.0 - 2.0* table2[i*N_s+j]);
    }}
    }
}




void compute_psi_kernels(float *psi, float *hfunc_cell)
{
    #pragma acc kernels present(psi[0:N_s*N_s],hfunc_cell[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=0; i<N_s; i++)
    {   
    #pragma acc loop independent
    for (int j=0;j<N_s;j++)
        {   psi[i*N_s+j] = 0.0;
            for (int n =0; n<Ntot; n++)
            {
            psi[i*N_s+j] += hfunc_cell[n*N_s*N_s+ i*N_s+j];
        }
        }
    }
    }
}



void compute_normal_grad_phi(float *table1, float *table2)
{   

    int ip; 
    int jp;
    int jm;
    int im;

    #pragma acc kernels present(table1[0:Ntot*N_s*N_s],table2[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int n=0; n<Ntot; n++)
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++){
        #pragma acc loop independent 
        for (int j=1;j<N_s-1;j++){

            ip = i+1;im = i-1;
            jp = j+1;jm = j-1;

            table1[n*N_s*N_s+i*N_s+j] = sqrt((1.0/dx/dx/4.0)* (
                            (table2[n*N_s*N_s+ip*N_s+j]-table2[n*N_s*N_s+im*N_s+j])*(table2[n*N_s*N_s+ip*N_s+j]-table2[n*N_s*N_s+im*N_s+j])+(table2[n*N_s*N_s+i*N_s+jp]-table2[n*N_s*N_s+i*N_s+jm])*(table2[n*N_s*N_s+i*N_s+jp]-table2[n*N_s*N_s+i*N_s+jm])
                            ));

            }}
    }
    }
}

void compute_normal_to_phi(float *phi, float *norm_grad_phi, float *J_x, float *J_y)
{
    #pragma acc kernels present(phi[0:Ntot*N_s*N_s], norm_grad_phi[0:Ntot*N_s*N_s], J_x[0:Ntot*N_s*N_s], J_y[0:Ntot*N_s*N_s])
    {
    
    //compute normal vector : grad(phi)/norm_grad(phi) BEWARE : using J_x and J_y for memory commodities
    
    #pragma acc loop independent
    for (int n=0; n<Ntot; n++)
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

    //cell1
    if (norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.00)
    {J_x[n*N_s*N_s+i*N_s+j] = (1.0/(2.0*dx))*(phi[n*N_s*N_s+ip*N_s+j]-phi[n*N_s*N_s+im*N_s+j])/norm_grad_phi[n*N_s*N_s+i*N_s+j];}
    else {J_x[n*N_s*N_s+i*N_s+j]=0.0;}
    if (norm_grad_phi[n*N_s*N_s+i*N_s+j]>0.00)
    {J_y[n*N_s*N_s+i*N_s+j] = (1.0/(2.0*dx))*(phi[n*N_s*N_s+i*N_s+jp]-phi[n*N_s*N_s+i*N_s+jm])/norm_grad_phi[n*N_s*N_s+i*N_s+j];}
    else {J_y[n*N_s*N_s+i*N_s+j]=0.0;}
    }
    }
    }
    }
}


void boundary_conditions_cells(float *table)
{
    #pragma acc kernels present(table[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int n=0; n<Ntot; n++)
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
        {
        table[n*N_s*N_s+i*N_s]=table[n*N_s*N_s+i*N_s+N_s-2];
        table[n*N_s*N_s+i*N_s+N_s-1]=table[n*N_s*N_s+i*N_s+1];
        
        table[n*N_s*N_s+i] = table[n*N_s*N_s+N_s*(N_s-2)+i];
        table[n*N_s*N_s+N_s*(N_s-1)+i] = table[n*N_s*N_s+N_s+i];
        }
    // 4 corners
    // #pragma acc parallel
    // {
    table[n*N_s*N_s+0] = table[n*N_s*N_s+N_s*(N_s-2)+N_s-2];
    table[n*N_s*N_s+N_s-1] = table[n*N_s*N_s+N_s*(N_s-2)+1];
    table[n*N_s*N_s+N_s*(N_s-1)] = table[n*N_s*N_s+N_s+N_s-2];
    table[n*N_s*N_s+N_s*(N_s-1)+N_s-1] = table[n*N_s*N_s+N_s+1];
    // }
    }
    }

    //same performance if kernels or parallel loops are used for this boundary conditions function

}

void boundary_conditions_dirichlet(float *lum, float side_value)
{ // Dirichlet boundary conditions for ECM :: ECM = 1.0 on the sides, for the lumen :: Lumen = 0.0 on the sides
    #pragma acc kernels present(lum[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {lum[i*N_s]=side_value; lum[i*N_s+N_s-1]=side_value;
    lum[i] = side_value; lum[N_s*(N_s-1)+i] = side_value;
    }
    // 4 corners
    lum[0] = 0.0;
    lum[N_s-1] = 0.0; 
    lum[(N_s-1)*N_s] = 0.0;
    lum[N_s*(N_s-1)+N_s-1] = 0.0;
    }
}

void boundary_conditions_lumen_and_ECM(float *lum, float *ECM)
{   // Dirichlet boundary conditions for ECM :: ECM = 1.0 on the sides, for the lumen :: Lumen = 0.0 on the sides

    #pragma acc kernels present(lum[0:N_s*N_s],ECM[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {lum[i*N_s]=0.0; lum[i*N_s+N_s-1]=0.0;
    lum[i] = 0.0; lum[N_s*(N_s-1)+i] = 0.0;
    
    ECM[i*N_s+0]=1.0; ECM[i*N_s+N_s-1]=1.0;
    ECM[i] = 1.0; ECM[N_s*(N_s-1)+i] = 1.0;
    }
    // 4 corners
    lum[0] = 0.0;
    lum[N_s-1] = 0.0; 
    lum[(N_s-1)*N_s] = 0.0;
    lum[N_s*(N_s-1)+N_s-1] = 0.0;
    ECM[0] = 1.0;
    ECM[N_s-1] = 1.0; 
    ECM[(N_s-1)*N_s] = 1.0;
    ECM[(N_s-1)*N_s+N_s-1] = 1.0;
    }
}


void boundary_conditions_Jx(float *table)
{
    #pragma acc kernels present(table[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    table[i*N_s+0]=table[i*N_s+N_s-2];
    table[i*N_s+N_s-1]=table[i*N_s+1];
    table[i] = 0.0; table[N_s*(N_s-1)+i] = 0.0;
    }
    // 4 corners
    table[0] = 0.0;
    table[N_s-1] = 0.0; 
    table[(N_s-1)*N_s] = 0.0;
    table[N_s*(N_s-1)+N_s-1] = 0.0;
    }
}

void boundary_conditions_lumen_Jy(float *table)
{
    #pragma acc kernels present(table[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    table[i*N_s+0]=table[i*N_s+N_s-2];
    table[i*N_s+N_s-1]=table[i*N_s+1];
    table[i] = -1.0; table[N_s*(N_s-1)+i] = 0.0;
    }
    // 4 corners
    table[0] = -1.0;
    table[N_s-1] = -1.0; 
    table[(N_s-1)*N_s] = 0.0;
    table[N_s*(N_s-1)+N_s-1] = 0.0;
    }
}



void boundary_conditions_periodic(float *ECM)
{   
    #pragma acc kernels present(ECM[0:N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    ECM[i*N_s]=ECM[i*N_s+N_s-2];
    ECM[i*N_s+N_s-1]=ECM[i*N_s+1];
    
    ECM[i] = ECM[N_s*(N_s-2)+i];
    ECM[N_s*(N_s-1)+i] = ECM[N_s+i];

    }
    // 4 corners
    ECM[0] = ECM[N_s*(N_s-2)+N_s-2];
    ECM[N_s-1] = ECM[N_s*(N_s-2)+1];
    ECM[(N_s-1)*N_s] = ECM[N_s+N_s-2];
    ECM[(N_s-1)*N_s+N_s-1] = ECM[N_s+1];

    }
}


void boundary_conditions_pi(float *pi_xx, float *pi_xy, float *pi_yy)
{
    //periodic boundary conditions
    #pragma acc kernels present(pi_xx[0:Ntot*N_s*N_s], pi_xy[0:Ntot*N_s*N_s], pi_yy[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    pi_xx[i*N_s]=pi_xx[i*N_s+N_s-2];
    pi_xx[i*N_s+N_s-1]=pi_xx[i*N_s+1];
    pi_yy[i*N_s]=pi_yy[i*N_s+N_s-2];
    pi_yy[i*N_s+N_s-1]=pi_yy[i*N_s+1];
    pi_xy[i*N_s]=pi_xy[i*N_s+N_s-2];
    pi_xy[i*N_s+N_s-1]=pi_xy[i*N_s+1];

    pi_xx[N_s*N_s+i*N_s]=pi_xx[N_s*N_s+i*N_s+N_s-2];
    pi_xx[N_s*N_s+i*N_s+N_s-1]=pi_xx[N_s*N_s+i*N_s+1];
    pi_yy[N_s*N_s+i*N_s]=pi_yy[N_s*N_s+i*N_s+N_s-2];
    pi_yy[N_s*N_s+i*N_s+N_s-1]=pi_yy[N_s*N_s+i*N_s+1];
    pi_xy[N_s*N_s+i*N_s]=pi_xy[N_s*N_s+i*N_s+N_s-2];
    pi_xy[N_s*N_s+i*N_s+N_s-1]=pi_xy[N_s*N_s+i*N_s+1];
    }
    }

    #pragma acc kernels present(pi_xx[0:Ntot*N_s*N_s], pi_xy[0:Ntot*N_s*N_s], pi_yy[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent
    for (int i=1;i<N_s-1;i++)
    {
    pi_xx[i] = pi_xx[(N_s-2)*N_s+i];
    pi_xx[(N_s-1)*N_s+i] = pi_xx[N_s+i];
    pi_yy[i] = pi_yy[(N_s-2)*N_s+i];
    pi_yy[(N_s-1)*N_s+i] = pi_yy[N_s+i];
    pi_xy[i] = pi_xy[(N_s-2)*N_s+i];
    pi_xy[(N_s-1)*N_s+i] = pi_xy[N_s+i];

    pi_xx[N_s*N_s+i] = pi_xx[N_s*N_s+(N_s-2)*N_s+i];
    pi_xx[N_s*N_s+(N_s-1)*N_s+i] = pi_xx[N_s*N_s+N_s+i];
    pi_yy[N_s*N_s+i] = pi_yy[N_s*N_s+(N_s-2)*N_s+i];
    pi_yy[N_s*N_s+(N_s-1)*N_s+i] = pi_yy[N_s*N_s+N_s+i];
    pi_xy[N_s*N_s+i] = pi_xy[N_s*N_s+(N_s-2)*N_s+i];
    pi_xy[N_s*N_s+(N_s-1)*N_s+i] = pi_xy[N_s*N_s+N_s+i];
    }
    }

    // 4 corners
    pi_xx[0] = pi_xx[(N_s-2)*N_s+N_s-2];
    pi_xx[N_s-1] = pi_xx[(N_s-2)*N_s+1];
    pi_xx[(N_s-1)*N_s] = pi_xx[N_s+N_s-2];
    pi_xx[(N_s-1)*N_s+N_s-1] = pi_xx[N_s+1];
    pi_yy[0] = pi_yy[(N_s-2)*N_s+N_s-2];
    pi_yy[N_s-1] = pi_yy[(N_s-2)*N_s+1];
    pi_yy[(N_s-1)*N_s] = pi_yy[N_s+N_s-2];
    pi_yy[(N_s-1)*N_s+N_s-1] = pi_yy[N_s+1];
    pi_xy[0] = pi_xy[(N_s-2)*N_s+N_s-2];
    pi_xy[N_s-1] = pi_xy[(N_s-2)*N_s+1];
    pi_xy[(N_s-1)*N_s] = pi_xy[N_s+N_s-2];
    pi_xy[(N_s-1)*N_s+N_s-1] = pi_xy[N_s+1];

    // 4 corners
    pi_xx[N_s*N_s+0] = pi_xx[N_s*N_s+(N_s-2)*N_s+N_s-2];
    pi_xx[N_s*N_s+N_s-1] = pi_xx[N_s*N_s+(N_s-2)*N_s+1];
    pi_xx[N_s*N_s+(N_s-1)*N_s] = pi_xx[N_s*N_s+N_s+N_s-2];
    pi_xx[N_s*N_s+(N_s-1)*N_s+N_s-1] = pi_xx[N_s*N_s+N_s+1];
    pi_yy[N_s*N_s+0] = pi_yy[N_s*N_s+(N_s-2)*N_s+N_s-2];
    pi_yy[N_s*N_s+N_s-1] = pi_yy[N_s*N_s+(N_s-2)*N_s+1];
    pi_yy[N_s*N_s+(N_s-1)*N_s] = pi_yy[N_s*N_s+N_s+N_s-2];
    pi_yy[N_s*N_s+(N_s-1)*N_s+N_s-1] = pi_yy[N_s*N_s+N_s+1];
    pi_xy[N_s*N_s+0] = pi_xy[N_s*N_s+(N_s-2)*N_s+N_s-2];
    pi_xy[N_s*N_s+N_s-1] = pi_xy[N_s*N_s+(N_s-2)*N_s+1];
    pi_xy[N_s*N_s+(N_s-1)*N_s] = pi_xy[N_s*N_s+N_s+N_s-2];
    pi_xy[N_s*N_s+(N_s-1)*N_s+N_s-1] = pi_xy[N_s*N_s+N_s+1];

}


void load_temp_fft(int cell_number, float *table, cufftComplex *temp_fft, int *xcenters, int *ycenters)
{   
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;    
    //cell number = 0 for cell1 1 for cell 2
    #pragma acc kernels present(table[0:Ntot*N_s*N_s], temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], xcenters[0:Ntot], ycenters[0:Ntot])
    {
        #pragma acc loop independent
        for(int i=0;i<2*zone_size-2;i++)
            {
            #pragma acc loop independent
            for(int j=0;j<2*zone_size-2;j++)
                {
                i_gridcell = i+1 + xcenters[cell_number]-zone_size;
                j_gridcell = j+1 + ycenters[cell_number]-zone_size;
                temp_fft[i*(2*zone_size-2)+j].x = table[cell_number*N_s*N_s+i_gridcell*N_s+j_gridcell];
                temp_fft[i*(2*zone_size-2)+j].y = 0.0;
                }
            }
    }
}

void load_temp_fft_rho(int cell_number, float *rho, cufftComplex *temp_fft, cufftComplex *temp_fft_real, int *xcenters, int *ycenters)
{   
    int i_gridcell = 0.0;
    int j_gridcell = 0.0;    
    #pragma acc kernels present(rho[0:Ntot*N_s*N_s], temp_fft[0:(2*zone_size-2)*(2*zone_size-2)], temp_fft_real[0:(2*zone_size-2)*(2*zone_size-2)], xcenters[0:Ntot], ycenters[0:Ntot])
    {
    #pragma acc loop independent
    for(int i=0;i<2*zone_size-2;i++){
    #pragma acc loop independent
    for(int j=0;j<2*zone_size-2;j++)
    {
            i_gridcell = i+1+xcenters[cell_number]-zone_size;
            j_gridcell = j+1+ycenters[cell_number]-zone_size;
            temp_fft[i*(2*zone_size-2)+j].x = rho[cell_number*N_s*N_s+i_gridcell*N_s+j_gridcell]*temp_fft_real[i*(2*zone_size-2)+j].x/((2*zone_size-2)*(2*zone_size-2));
            temp_fft[i*(2*zone_size-2)+j].y = 0.0;    
    }}
    }
}

void compute_fft_forward(cufftHandle planf, cufftComplex *temp_fft, cufftComplex *fft_table)
{   
    #pragma acc host_data use_device(temp_fft,fft_table)
    {
    cufftExecC2C(planf, temp_fft, fft_table, CUFFT_FORWARD);           
    }
}

void compute_fft_backward(cufftHandle planr, cufftComplex *fft_table, cufftComplex *temp_table)
{
    #pragma acc host_data use_device(fft_table, temp_table)
        {
          cufftExecC2C(planr, fft_table, temp_table, CUFFT_INVERSE);
        }
}

void copy_cell_tables(float *phase, float *phasen)
{
    #pragma acc kernels present(phase[0:Ntot*N_s*N_s], phasen[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent 
    for (int i=0;i<N_s;i++)
    {
        #pragma acc loop independent
        for (int j=0; j<N_s; j++)
        {
            phase[i*N_s+j] = phasen[i*N_s+j];
            phase[N_s*N_s+i*N_s+j] = phasen[N_s*N_s+i*N_s+j];
        }
    }
    }
}

void copy_cell_tables_1_by_1(int n, float *phase, float *phasen)
{
    #pragma acc kernels present(phase[0:Ntot*N_s*N_s], phasen[0:Ntot*N_s*N_s])
    {
    #pragma acc loop independent 
    for (int i=0;i<N_s;i++)
    {
        #pragma acc loop independent
        for (int j=0; j<N_s; j++)
        {
            phase[n*N_s*N_s+i*N_s+j] = phasen[n*N_s*N_s+i*N_s+j];
        }
    }
    }
}

void copy_tables_ECM_or_lumen(float *phase, float *phasen)
{
    #pragma acc kernels present(phase[0:N_s*N_s], phasen[0:N_s*N_s])
    {
    #pragma acc loop independent 
    for (int i=0;i<N_s;i++)
    {
        #pragma acc loop independent
        for (int j=0; j<N_s; j++)
        {
            phase[i*N_s+j] = phasen[i*N_s+j];
        }
    }
    }
}
    