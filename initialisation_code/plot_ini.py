# -*- coding: utf-8 -*-
"""
@author: Tristan Guyomar (Riveline Lab - IGBMC - University of Strasbourg)
"""
import numpy as np
import matplotlib.pyplot as plt 
import os 


def how_many_cells(path):

    count = 0
    names = os.listdir(path)
    for name in names:
        if '.bin' in name:
            if 'ini_cell_' in name:
                if 'before' not in name:
                    count +=1
    return(count)

ori_path = os.path.abspath(os.getcwd())


Ncells = how_many_cells(ori_path)
foldername = 'data_ini_to_plot/'
print('Number of cells = ', Ncells)

for n in range(Ncells):
    
    phi_gpu = np.genfromtxt(foldername+'ini_cell_'+str(n+1)+'_before_reshaping.dat')
    N = int(np.sqrt(len(phi_gpu)))
    print(int(N))
    
    phi_gpu = np.genfromtxt(foldername+'ini_cell_'+str(n+1)+'_before_reshaping.dat').reshape((N,N))

    rho_gpu = np.genfromtxt(foldername+'ini_rho_'+str(n+1)+'_before_reshaping.dat').reshape((N,N))

    plt.figure(1)
    plt.imshow(phi_gpu)
    plt.savefig('all_plots_ini/comparison_before_after_reshaping/ini_phase_'+str(n+1)+'_before_reshaping.png')

    plt.figure(2)
    plt.imshow(rho_gpu)
    plt.savefig('all_plots_ini/comparison_before_after_reshaping/ini_rho_'+str(n+1)+'_before_reshaping.png')

    phi_gpu = np.genfromtxt(foldername+'ini_cell_'+str(n+1)+'.dat')
    N = int(np.sqrt(len(phi_gpu)))
    print(int(N))
    phi_gpu = np.genfromtxt(foldername+'ini_cell_'+str(n+1)+'.dat').reshape((N,N))

    rho_gpu = np.genfromtxt(foldername+'ini_rho_'+str(n+1)+'.dat').reshape((N,N))

    plt.figure(1)
    plt.imshow(phi_gpu)
    plt.savefig('all_plots_ini/comparison_before_after_reshaping/ini_phase_'+str(n+1)+'.png')

    plt.figure(2)
    plt.imshow(rho_gpu)
    plt.savefig('all_plots_ini/comparison_before_after_reshaping/ini_rho_'+str(n+1)+'.png')


phi_gpu = np.genfromtxt(foldername+'ini_lumen.dat')
N = int(np.sqrt(len(phi_gpu)))
phi_gpu = np.genfromtxt(foldername+'ini_lumen.dat').reshape((N,N))

rho_gpu = np.genfromtxt(foldername+'ini_ECM.dat').reshape((N,N))



plt.figure(1)
plt.imshow(phi_gpu)
plt.savefig('all_plots_ini/initial_lumen.png')

plt.figure(2)
plt.imshow(rho_gpu)
plt.savefig('all_plots_ini/initial_ECM.png')

print("I have saved the picture of the initial conditions for ECM and lumen ! \n")