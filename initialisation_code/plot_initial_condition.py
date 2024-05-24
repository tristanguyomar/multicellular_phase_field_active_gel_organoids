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
print('Starting to plot the initial condition.')
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

names = os.listdir(path)
cell_names = []
rho_names = []

for name in names:
    if 'reshaping' not in name:
        if 'cell' in name:
            cell_names.append(name)
        if 'rho' in name:
            rho_names.append(name)


N = int(np.sqrt(len(np.genfromtxt(path+names[0]))))
ncells = how_many_cells(ori_path)
full_img = np.zeros((N,N))

for i in range(0,ncells):
    rho = np.genfromtxt(path+rho_names[i])
    N = int(np.sqrt(len(rho)))
    rho = rho.reshape(N,N)
    for ii in range(N):
            for jj in range(N):
                if rho[ii,jj]>full_img[ii,jj]:
                    full_img[ii,jj] = rho[ii,jj]


fig,ax = plt.subplots() # initialise la figure
#[400:1200,400:1200]
ax.imshow(full_img, cmap = plt.cm.CMRmap, interpolation='none', aspect='auto', vmin=np.nanmin(full_img), vmax=np.nanmax(full_img))
plt.title('Active gel initial condition')
ax.set_aspect('equal')

name = 'all_cortices_initial.png'
os.chdir(ori_path)
plt.axis('off')
plt.savefig(ori_path + '/all_plots_ini/'+name, bbox_inches='tight', pad_inches = 0, dpi = 1000)


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

N = int(np.sqrt(len(np.genfromtxt(path+names[0]))))

ncells = how_many_cells(ori_path)

image = np.zeros((N,N))


lumen_name = path+'ini_lumen.dat'
lumen = np.genfromtxt(lumen_name).reshape(N,N)
lumen = lumen*lumen*(3-2*lumen)
image[lumen>=0.5] = 0.4
ecm_name = path+'ini_ECM.dat'
ecm = np.genfromtxt(ecm_name).reshape(N,N)
ecm = ecm*ecm*(3-2*ecm)
image[ecm>=0.5] = 1.0


full_img1 = np.zeros((N,N))

for cell_name in cell_names:
    cell = np.genfromtxt(cell_name).reshape(N,N)
    cell = cell*cell*(3-2*cell)
    image[cell>=0.5] = 0.6

    phi = np.genfromtxt(cell_name).reshape(N,N)
    phi[phi<0.5] = 0.0
    full_img1 += phi

image[image<0.4] = 0.1
    

fig,ax = plt.subplots() # initialise la figure
ax.imshow(full_img1, cmap = plt.cm.viridis, interpolation='none', aspect='auto', vmin=np.nanmin(full_img1), vmax=np.nanmax(full_img1))
plt.title('Cells initial conditions')
ax.set_aspect('equal')

name = 'all_cells_initial.png'
os.chdir(ori_path)
plt.axis('off')
plt.savefig(ori_path + '/all_plots_ini/'+name, bbox_inches='tight', pad_inches = 0, dpi = 1000)



fig, ax = plt.subplots()
ax.imshow(image,cmap = newcmp, vmin =0, vmax = 1.0, interpolation = 'nearest') 
ax.set_aspect('equal')
# plt.colorbar()
plt.axis('off')
name = 'all_phases_initial.png'
os.chdir(ori_path)
plt.title('Cells, lumen and ECM initial conditions')
plt.savefig(ori_path + '/all_plots_ini/'+name, bbox_inches='tight', pad_inches = 0, dpi = 1000)
