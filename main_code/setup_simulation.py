# -*- coding: utf-8 -*-
"""
Created on Wed Jun 6 2022

@author: Riveline LAB
"""

import os
import jinja2
import numpy as np
from datetime import date
import shutil


constants = {'ncells' : np.array([6]), 'epsilon_bas' : np.array([1.0]), 'epsilon_lat' : np.array([1.0]), 'epsilon_ap' : np.array([1.0]), 'a_ort_bas' : np.array([100.0]), 'a_ort_lat' : np.array([100.0]), 'a_ort_ap' : np.array([100.0]), 'b_bas' : np.array([1.0]), 'b_lat' : np.array([1.0]), 'b_ap' : np.array([1.0]), 'gen_ap' : np.array([1.0]), 'deg_ap': np.array([100.0]), 'viscosity_ap' : np.array([1.0]),  'viscosity_bas' : np.array([1.0]),  'viscosity_lat' : np.array([1.0]), 'friction' : np.array([1.0]), 'eta' : np.array([0.03]),  'xi_lumen' : np.array([0.15]), 'eta_lum_cells' : np.array([0.0]), 'alpha_ECM' : np.array([0.002]), 'eta_ecm_cells' : np.array([0.001])}

def compute_value_len(dic):

    len_c = []
    keys = dic.keys()
    for c in dic.keys():
        a = dic[c]
        if type(a) == np.ndarray :
            len_c.append(len(a))
        else :
            len_c.append(0)
    return(len_c)

def create_constant_dictionnary(full_constants_dic):

    list_dic = []    
    dic = full_constants_dic.copy()
    keys = list(dic.keys())
    len_c = compute_value_len(full_constants_dic)
    arg_len_c = np.argsort(len_c)[::-1]

    if len_c[arg_len_c[0]]>0:

        key = keys[arg_len_c[0]]
        value = full_constants_dic[key]
        for v in value :
            dic[key] = v
            if sum(compute_value_len(dic))==0:
                list_dic.append(dic.copy())
            else:
                new_list_dic = create_constant_dictionnary(dic)
                list_dic+=new_list_dic

    return(list_dic)

def create_savepath(const_dic):
    list_names = ['ncells', 'epsilon', 'a_ort','b','ratio_a_bl','gen_lat','deg_lat','gen_ap','deg_ap','gen_bas','deg_bas', 'turnover_ratio', 'viscosity','friction','eta','xi_lumen','eta_lum_cells','alpha_ECM','eta_ecm_cells']
    ind_to_keep = [0, 1, 2, 3, 4, 5, 6, 12, 13]
    list_names_to_keep = [list_names[i] for i in ind_to_keep]

    savepath = 'test_'
    keys = const_dic.keys()
    for c in keys:
        if c in list_names_to_keep:
            savepath+= c + '_' + str(round(const_dic[c],1)) + '_'
    
    savepath = savepath[:-1]
    return(savepath)


def create_constant_file(const_dic, save_path):

    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    TEMPLATE_FILE = "constants_template.cpp"
    template = templateEnv.get_template(TEMPLATE_FILE)
    output_constants = template.render(**const_dic)  

    if not os.path.exists(save_path):
        with open(save_path, 'w') as f :
            f.write(output_constants)
    
    return()

def copy_files_extension(origin_path, dest_path, extension):

    for filename in os.listdir(origin_path):
        if filename.endswith(extension):
            shutil.copy( origin_path + filename, dest_path)
    return()

today = str(date.today())
today = today.split('-')
today = '_'.join(today)
list_dic = create_constant_dictionnary(constants)

DIRECTORY = []
NAMES = []
for i,dic in enumerate(list_dic):
    directory = today+'_'+str(i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)

    #savepath =  today+'_'+str(i)+'/'+create_savepath(dic)+'.cpp'
    savepath_constants =  today+'_'+str(i)+'/constants.cpp'

    #create_constant_file(dic, savepath)
    create_constant_file(dic, savepath_constants)

    copy_files_extension(os.path.abspath(os.getcwd())+'/', directory, '.cpp')
    copy_files_extension(os.path.abspath(os.getcwd())+'/', directory, '.h')
    copy_files_extension(os.path.abspath(os.getcwd())+'/', directory, '.bin')
    copy_files_extension(os.path.abspath(os.getcwd())+'/', directory, '.py')
    copy_files_extension(os.path.abspath(os.getcwd())+'/', directory, '.sh')
    
    
    os.makedirs(directory+'/data_int')
    os.makedirs(directory+'/data_test')
    os.makedirs(directory+'/all_plots')
    os.makedirs(directory+'/all_plots/final_step')

    DIRECTORY.append(directory)
    NAMES.append(create_savepath(dic)+'.out')

i=0
with open('launch_all.sh', 'w') as f :
    for directory in DIRECTORY:
        f.write('cd '+directory+' \n')
        f.write('sbatch -o ' +NAMES[i] +' run_cell_all.sh \n')
        f.write('cd .. \n')
        i+=1

i=0
with open('launch_plot_files.sh', 'w') as f :
    for directory in DIRECTORY:
        f.write('cd '+directory+' \n')
        f.write('sbatch -o plot.out plot_files.sh \n')
        f.write('cd .. \n')
        i+=1

i=0
with open('launch_plot_final.sh', 'w') as f :
    for directory in DIRECTORY:
        f.write('cd '+directory+' \n')
        f.write('sbatch -o plot_final.out plot_final.sh \n')
        f.write('sbatch -o plot_movie.out plot_files_only_lumen_ecm_dynamics.sh \n')
        f.write('cd .. \n')
        i+=1