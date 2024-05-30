#  multicellular_phase_field_active_gel_organoids
<img src="https://github.com/tristanguyomar/multicellular_phase_field_active_gel_organoids/blob/main/github_figure.png" width="800">
<hr/>

The `multicellular_phase_field_active_gel_organoids` library is an implementation of the multicellular phase-field model coupled to an active gel description of the cellular cortex that simulate cell and organoid shape as described in ...

## Overview

### The multicellular phase-field model coupled with an active gel description of the cell cortex

To compile this code, you need to have the PGI compiler installed. It is available at http://www.pgroup.com/ or within the NVC compiler available at : https://developer.nvidia.com/nvidia-hpc-sdk-233-downloads

## Usage

Before starting, locate all the files contained in the folders "main_code", "functions", "initialisation_code" to the same folder and run all the following instructions from this directory.

### I.a. Setting up the initial condition for the simulation

To set up the initial condition for the simulation, you will need to change all simulation parameters by updating the 'constants_template.cpp' file.
Here, you will be able to choose the grid size, the time step and spatial grid step.
Then, you can update the 'setup_simulation_ini.py' file to change the parameters that are labeled between '{{...}}' in the constants_template.cpp file.

When the parameters are chosen, run :
```sh
python setup_simulation_ini.py
```
This will create a folder for each set of parameters that have been specified in the 'setup_simulation_ini.py' file.

### I.b. Compiling the initialisation code
Within each created folder run :

```sh
nvc++ -c constants.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o constants.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c declare_tables.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o declare_tables.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c initialisation_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o initialisation_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c useful_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o useful_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c cell_phase_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o cell_phase_functions.o -I${CUDAPATH}/include -lcudart -lcufft

nvc++ -c prepare_initial_files.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o prepare_initial_files.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ constants.o declare_tables.o initialisation_functions.o useful_functions.o cell_phase_functions.o prepare_initial_files.o -v -ta=tesla -tp=px -Mcuda -acc -Minfo=all,accel -Mvect=levels:5 -o prepare_initial_files -I${CUDAPATH}/include -lcudart -lcufft
```

### I.c. Running the initialisation code
Within each created folder run :
```sh
./prepare_initial_files
```

### I.c. Plotting the initial configuration

To plot the initial condition in various ways: you can use the code :
```sh
python plot_initial_condition.py '/data_ini_to_plot/'
python plot_ini.py
```
Then, you will have to retrieve the initial conditions files : 'ini_cell_1.bin' from the created folder and put them in the same directory as the 'setup_simulation.py' file.

### II.a Setting up the main code parameters

Here you will have to choose the number of steps that you want to compute and the number of files you want to save. Be careful that since it is a GPU accelerated code, the data is only retrieved at the end of the number of steps that you precise, that is why the code was adpated to be able to use the same compile code in sequential loops and retrieve computed data to able the user to follow the evolution of the simulation overtime. These parameters are to be changed in the 'constants_template.cpp' file.
Typical working number of steps was 100001 and it was runned about 25 times in 24 hours with a saving step every 10000 steps.  

When the parameters are chosen, run :
```sh
python setup_simulation.py
```
This command will create a folder for each specified set of parameters. 

### II.a Compiling the main code
Then run in each of the newly created subfolders:

```sh
nvc++ -c constants.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o constants.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c declare_tables.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o declare_tables.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c declare_fft_tables.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o declare_fft_tables.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c initialisation_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o initialisation_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c useful_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o useful_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c rho_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o rho_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c cell_phase_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o cell_phase_functions.o -I${CUDAPATH}/include -lcudart -lcufft

nvc++ -c run_with_active_gels.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o run_with_active_gels.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ constants.o declare_tables.o declare_fft_tables.o initialisation_functions.o useful_functions.o rho_functions.o cell_phase_functions.o run_with_active_gels.o -v -ta=tesla -tp=px -Mcuda -acc -Minfo=all,accel -Mvect=levels:5 -o run_with_active_gels -I${CUDAPATH}/include -lcudart -lcufft
```

### II.a Running the main code

As before, run within each created folder:
```sh
x=0
while [ $x -le 29 ]
do
    ./run_with_active_gels
    python rename_files.py '/data_test/'
    x=$(($x + 1))
done
```
Here you can choose the number of time you want to run the main code. The code 'rename_files.py' is needed to be able to overwrite files and proper saving of the data.

## Authors

* Tristan Guyomar - University of Strasbourg

## Dependencies

- nvcc
- pgi compiler
- python 

## License
"multicellular_phase_field_active_gel_organoids" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
