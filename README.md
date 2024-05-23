#  multicellular_phase_field_active_gel_organoids
<img src="https://github.com/tristanguyomar/multicellular_phase_field_active_gel_organoids/blob/main/github_figure.png" width="800">
<hr/>

The `multicellular_phase_field_active_gel_organoids` library is an implementation of the multicellular phase-field model coupled to an active gel description of the cellular cortex that simulate cell and organoid shape as described in ...

## Overview

### The multicellular phase-field model coupled with an active gel description of the cell cortex

To compile this code, you need to have the PGI compiler installed. It is available at http://www.pgroup.com/.
## Usage
```sh

# Setting up the initial condition for the simulation


# Code compile
nvcc -O3 -DSFMT_MEXP=19937 src/mcpf_2d_usc.cu src/SFMT.c -o run_simulation -std=c++11

# Runnning simulation from 8 cells: 
./run.sh test 200 0.18
# The first argument represents the parameter name, the second one is the cell growth time scale, and the third one is the rumen pressure.

```

## Authors

* Tristan Guyomar - University of Strasbourg

## Dependencies

- nvcc
- pgi compiler
- python 

## License
"multicellular_phase_field_active_gel_organoids" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
