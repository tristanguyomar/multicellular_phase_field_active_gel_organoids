#  multicellular_phase_field_active_gel_organoids
<img src="https://github.com/tristanguyomar/multicellular_phase_field_active_gel_organoids/blob/main/github_figure.png" width="800">
<hr/>

The `multicellular_phase_field_active_gel_organoids` library is an implementation of the multicellular phase-field model coupled to an active gel description of the cellular cortex that simulate cell and organoid shape as described in ...

## Overview

### The multicellular phase-field model coupled with an active gel description of the cell cortex

To compile this code, you need to have the PGI compiler installed. It is available at http://www.pgroup.com/ or within the NVC compiler available at : https://developer.nvidia.com/nvidia-hpc-sdk-233-downloads

## Usage

### Setting up the initial condition for the simulation

To set up the initial condition for the simulation, you will need to change all simulation parameters by updating the 'constants_template.cpp' file.
Here, you will be able to choose the grid size, the time step and spatial grid step.
Then, you can update the 'setup_simulation_ini.py' file to change the parameters that are labeled between '{{...}}' in the constants_template.cpp file.

When the parameters are chosen, run :
```sh
python setup_simulation_ini.py
```
This will create a folder for each set of parameters that have been specified in the 'setup_simulation_ini.py' file.

### Compiling the initialisation code

```sh
nvc++ -c constants.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o constants.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c declare_tables.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o declare_tables.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c initialisation_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o initialisation_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c useful_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o useful_functions.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ -c cell_phase_functions.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o cell_phase_functions.o -I${CUDAPATH}/include -lcudart -lcufft

nvc++ -c prepare_initial_files.cpp -v -fastsse -lm -Mipa=fast -ta=tesla -tp=px -Mcuda -acc -Minfo=accel -Mvect=levels:5 -o prepare_initial_files.o -I${CUDAPATH}/include -lcudart -lcufft
nvc++ constants.o declare_tables.o initialisation_functions.o useful_functions.o cell_phase_functions.o prepare_initial_files.o -v -ta=tesla -tp=px -Mcuda -acc -Minfo=all,accel -Mvect=levels:5 -o prepare_initial_files -I${CUDAPATH}/include -lcudart -lcufft
```
# Runnning simulation to save initial conditions:
./launch_all.sh



## Authors

* Tristan Guyomar - University of Strasbourg

## Dependencies

- nvcc
- pgi compiler
- python 

## License
"multicellular_phase_field_active_gel_organoids" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
