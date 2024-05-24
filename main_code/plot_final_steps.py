# -*- coding: utf-8 -*-
"""
@author: Tristan Guyomar (Riveline Lab - IGBMC - University of Strasbourg)
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
import matplotlib.animation as animation
import sys 

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.close('all')
ori_path = os.path.abspath(os.getcwd())
print(ori_path)
path_input = sys.argv[1]
path = ori_path+path_input
os.chdir(path)

def how_many_cells(path):

    count = 0
    names = os.listdir(path)
    for name in names:
        if '.bin' in name:
            if 'ini_cell_' in name:
                if 'before' not in name:
                    count +=1
    return(count)



def find_files(names, string):
    phi_names = []
    for name in names:
        if string in name:
            phi_names.append(name)
    return(phi_names)

def find_and_order_lumen_ECM_files(path):
    
    lumen_names = []
    ECM_names = []
    time_lumen = []
    time_ECM = []
    names = os.listdir(path)
    for name in names:
        if 'lumen' in name:
            lumen_names.append(name)
            time_lumen.append(int(name.split('_')[-1].split('.dat')[0]))
        if 'ECM' in name:
            ECM_names.append(name)
            time_ECM.append(int(name.split('_')[-1].split('.dat')[0]))
    
    sorted_lumen_names = [lumen_names[i] for i in np.argsort(time_lumen)]
    sorted_ECM_names = [ECM_names[i] for i in np.argsort(time_ECM)]
    
    return(sorted_lumen_names, sorted_ECM_names)

def order_sequence_cell(names, label):
    
    cell_names = []
    time = []
    for name in names:
        if str(label) == name.split('_')[2]:
            cell_names.append(name)
            time.append(int(name.split('_')[-1].split('.dat')[0]))
            
    sorted_cell_names = [cell_names[i] for i in np.argsort(time)]
    return(sorted_cell_names, np.sort(time))

def animate(i):

    img.set_array(snapshots[i])
    
    plt.title('Step = '+str(time_serie[i]))
    plt.axis('scaled')
    return[img]

def all_cell_names_single_timepoint(names, t):
    
    all_cell_names = []
    
    for name in names:
        if 'phi' in name:
            if 'step_'+t+'.dat' in name:
                all_cell_names.append(name)
    
    return(all_cell_names)


names = os.listdir(path)

rho_names = find_files(names, 'rho')

N = int(np.sqrt(len(np.genfromtxt(path+names[0]))))

ncells = how_many_cells(ori_path)
nt = int(len(rho_names)/(ncells))
full_img = np.zeros((N,N))

for i in range(1,ncells+1):
    print(i)
    cell_names,time = order_sequence_cell(rho_names, i)
    print(cell_names[-1])
    rho = np.genfromtxt(path+ cell_names[-1])
    N = int(np.sqrt(len(rho)))
    rho = rho.reshape(N,N)
    for ii in range(N):
            for jj in range(N):
                if rho[ii,jj]>full_img[ii,jj]:
                    full_img[ii,jj] = rho[ii,jj]


fig,ax = plt.subplots() # initialise la figure
#[400:1200,400:1200]
ax.imshow(full_img, cmap = plt.cm.CMRmap, interpolation='none', aspect='auto', vmin=np.nanmin(full_img), vmax=np.nanmax(full_img))
plt.title('Timestep = '+str(time[-1]))
ax.set_aspect('equal')

name = 'all_plots/final_step/all_cortices_final.png'
os.chdir(ori_path)
plt.axis('off')
plt.savefig(name, bbox_inches='tight', pad_inches = 0, dpi = 1000)


fig,ax = plt.subplots() # initialise la figure
#[400:1200,400:1200]
ax.imshow(full_img, cmap = 'seismic', interpolation='none', aspect='auto')
plt.title('Timestep = '+str(time[-1]))
ax.set_aspect('equal')

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))

black = np.array([0, 0, 0.0, 1])
newcolors[:256//4, :] = black

blue = np.array([0.0784313725490196, 0.6, 0.9803921568627451, 1])
newcolors[256//4:2*(256//4), :] = blue #so lumen values > 0.5 are set to be 0.4


red = np.array([0.9803921568627451, 0.0196078431372549, 0, 1])
newcolors[2*(256//4):3*(256//4), :] = red #so cell values > 0.5 are set to be 0.6 

yellow = np.array([1, 0.9215686274509803, 0.050980392156862744, 1])
newcolors[3*(256//4):, :] = yellow #so ecm values > 0.5 are set to be 1

newcmp = ListedColormap(newcolors)


os.chdir(path)
lumen_names, ECM_names = find_and_order_lumen_ECM_files(path)
names = os.listdir(path)
phi_names = find_files(names, 'phi')


N = int(np.sqrt(len(np.genfromtxt(path+names[0]))))

ncells = how_many_cells(ori_path)
nt = int(len(phi_names)/(ncells))

dx = 0.01
image = np.zeros((N,N))

times = np.sort(np.unique([int(e.split('_')[-1].split('.')[0]) for e in phi_names]))
time_labels = [str(e) for e in times]

i = -1
lumen_name = path+lumen_names[i]
print(lumen_name)
lumen = np.genfromtxt(lumen_name).reshape(N,N)
lumen = lumen*lumen*(3-2*lumen)
image[lumen>=0.5] = 0.4
ecm_name = path+ECM_names[i]
print(ecm_name)
ecm = np.genfromtxt(ecm_name).reshape(N,N)
ecm = ecm*ecm*(3-2*ecm)
image[ecm>=0.5] = 1.0

time_label = time_labels[i]
all_cell_names = all_cell_names_single_timepoint(names, time_label)

full_img1 = np.zeros((N,N))

for cell_name in all_cell_names:
    print(cell_name)
    cell = np.genfromtxt(cell_name).reshape(N,N)
    cell = cell*cell*(3-2*cell)
    image[cell>=0.5] = 0.6

    phi = np.genfromtxt(cell_name).reshape(N,N)
    phi[phi<0.5] = 0.0
    full_img1 += phi

image[image<0.4] = 0.1
    

fig,ax = plt.subplots() # initialise la figure
ax.imshow(full_img1, cmap = plt.cm.viridis, interpolation='none', aspect='auto', vmin=np.nanmin(full_img1), vmax=np.nanmax(full_img1))
plt.title('Timestep = '+str(time[-1]))
ax.set_aspect('equal')

name = 'all_plots/final_step/all_cells_final.png'
os.chdir(ori_path)
plt.axis('off')
plt.savefig(name, bbox_inches='tight', pad_inches = 0, dpi = 1000)



fig, ax = plt.subplots()
ax.imshow(image,cmap = newcmp, vmin =0, vmax = 1.0, interpolation = 'nearest') 
ax.set_aspect('equal')
# plt.colorbar()
plt.axis('off')
name = 'all_plots/final_step/all_phases.png'
os.chdir(ori_path)
plt.title('Timestep = '+str(time[-1]))
plt.savefig(name, bbox_inches='tight', pad_inches = 0, dpi = 1000)