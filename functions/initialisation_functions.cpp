#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "constants.h"
#include "declare_tables.h"

void ini_phi_files(float *phi)
{   
    char filename[64];
    for(int n=0;n<N_ini;n++){
    snprintf(filename, sizeof(filename), "ini_cell_%d.bin", n+1);

    std::ifstream file(filename, std::ios::binary);
    for (int i=0;i<N_s*N_s;i++)
    {file.read(reinterpret_cast<char*>(&phi[n*N_s*N_s+(i/N_s*N_s)+i%N_s]), sizeof(&phi[n*N_s*N_s+(i/N_s*N_s)+i%N_s]));}}

    /*
    //if the input file is a .dat

    FILE *input=NULL;
    input = fopen(filename, "r");
    for (int i=0;i<N_s*N_s;i++)
    {    
    fscanf(input,"%f\n",&phi[n*N_s*N_s+(i/N_s*N_s)+i%N_s]);
    }
    fclose(input); 
    }
    */
}



void ini_phin(float *phin, int n)
{
    for (int i = 0; i<N_s;i++){for (int j=0;j<N_s;j++){phin[n*N_s*N_s+i*N_s+j] = 0.0;}}

}

void ini_rho_files(float *rho)
{   
    char filename[64];
    for(int n=0;n<N_ini;n++){
    snprintf(filename, sizeof(filename), "ini_rho_%d.bin", n+1);

    std::ifstream file(filename, std::ios::binary);
    for (int i=0;i<N_s*N_s;i++)
    {file.read(reinterpret_cast<char*>(&rho[n*N_s*N_s+(i/N_s*N_s)+i%N_s]), sizeof(&rho[n*N_s*N_s+(i/N_s*N_s)+i%N_s]));}}

    /*    
    //if the input file is a .dat
    
    FILE *input=NULL;
    input = fopen(filename, "r");
    for (int i=0;i<N_s*N_s;i++)
    {    
    fscanf(input,"%f\n",&rho[n*N_s*N_s+(i/N_s*N_s)+i%N_s]);
    }
    fclose(input); 
    }
    */
}


void ini_rho(float *rho, float *phi, float rho0, float tol, int n)
{
    for (int i=0;i<N_s;i++)
    {for (int j=0;j<N_s;j++)
    {rho[n*N_s*N_s+i*N_s+j] = rho0*exp(-1.0/(4.0*phi[n*N_s*N_s+i*N_s+j]*phi[n*N_s*N_s+i*N_s+j]*(3.0-2.0*phi[n*N_s*N_s+i*N_s+j])*(1.0-phi[n*N_s*N_s+i*N_s+j]*phi[n*N_s*N_s+i*N_s+j]*(3.0-2.0*phi[n*N_s*N_s+i*N_s+j]))+tol)+1.0/(1.0+tol));
    if(rho[n*N_s*N_s+i*N_s+j] < rho0*exp(-1.0/(2.0*tol)+1.0/(1.0+tol))) {rho[n*N_s*N_s+i*N_s+j] = 0.0;}
}
}
}

void ini_rhon(float *rhon, int n)
{
    for (int i=0;i<N_s;i++){for (int j=0;j<N_s;j++){rhon[n*N_s*N_s+i*N_s+j] = 0.0;}}

}


void ini_ECM_lumen(float *lum, float *ECM, float *phi)
{   
    float rcyst;
    rcyst = N_ini*0.9*R/dx/M_PI;

    float all_cells = 0.0;
    float d = 0.0;
    for (int i=0; i<N_s;i++){for (int j=0;j<N_s;j++){
        all_cells = 0.0;
        for (int n=0; n<N_ini; n++)
        {
        all_cells += phi[n*N_s*N_s+N_s*i+j]; 
        }
    if (1.0-all_cells>0.0)
    {   d = sqrt((i-N_s/2)*(i-N_s/2)+(j-N_s/2)*(j-N_s/2));
        if (d < rcyst)
        {
            lum[i*N_s+j] = 1.0-all_cells;
        }
        else
        {
            ECM[i*N_s+j] = 1.0-all_cells;
        }
    }
    else
    {
        lum[i*N_s+j] = 0.0;
        ECM[i*N_s+j] = 0.0;
    }
    }}
}

void ini_phi(float *phi)
{   
    float xcenter;
    float ycenter; 
    float rcyst;
    rcyst = N_ini*0.9*R/dx/M_PI;

    if (N_ini>1)
    {
    for (int n=0; n<N_ini; n++)
    {   
 
        xcenter = N_s/2 + rcyst*cos(n*2.0*M_PI/N_ini);
        ycenter = N_s/2 + rcyst*sin(n*2.0*M_PI/N_ini);

    for (int i=0;i<N_s; i++)
    {
        for (int j=0;j<N_s; j++)
        {
            phi[n*N_s*N_s+i*N_s+j] = 1.0/(1.0+exp((dx*sqrt((i-xcenter)*(i-xcenter)+(j-ycenter)*(j-ycenter))-R0)/wi));
 
        }
    }
    }
    }
    else
    {
        xcenter = N_s/2;
        ycenter = N_s/2;

        for (int i=0;i<N_s; i++){
        for (int j=0;j<N_s; j++)
        {
            phi[i*N_s+j] = 1.0/(1.0+exp((dx*sqrt((i-xcenter)*(i-xcenter)+(j-ycenter)*(j-ycenter))-R0)/wi));
 
        }
    }
    }
}




void ini_lumen_files(float *lum)
{   
    char filename[64];
    sprintf(filename,"ini_lumen.bin");

    std::ifstream file(filename, std::ios::binary);
    for (int i=0;i<N_s*N_s;i++)
    {file.read(reinterpret_cast<char*>(&lum[i]), sizeof(&lum[i]));}
}



void ini_ECM_files(float *ECM)
{   
    char filename[64];
    sprintf(filename,"ini_ECM.bin");

    std::ifstream file(filename, std::ios::binary);
    for (int i=0;i<N_s*N_s;i++)
    {file.read(reinterpret_cast<char*>(&ECM[i]), sizeof(&ECM[i]));}

    /*
    //if the input file is a .dat
    FILE *input=NULL;

    input = fopen(filename, "r");
    for (int i=0;i<N_s*N_s;i++)
    {
    fscanf(input,"%f\n",&ECM[i]);
    }
    fclose(input); 
    */
}
