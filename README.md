# sensing-normalized-results

This repository contains code and scripts to recreate the results of the manuscript "Normalized Neural Representations of Natural Odors".
The folder `src` contains several python packages that contain functions and classes for studying normalized and unnormalized odor representations.
The folder `figures` contains scripts that use these functions and classes to recreate the figures of the publication.
They thereby also provide some examples of how to use the respective classes.

The python code should run under version 2 and 3 if the following additional packages are installed:

To run the code, the following python packages might be nes

Package     | Usage
------------|-------------------------------------------
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
six         | Compatibility layer to support python 2 and 3
cma         | For optimizations using CMA-ES
numba       | For creating compiled code for faster processing
nose        | For parallel testing
simanneal   | Simulated annealing algorithm published at https://github.com/perrygeo/simanneal
py-utils    | A collection of python functions published at https://github.com/david-zwicker/py-utils

Some of these package might only be required for a small subset of the code.
