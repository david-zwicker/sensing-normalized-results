# Sensing #

This repository contains code for simulations of the sensing project.

Necessary python packages:

Package     | Usage                                      
------------|-------------------------------------------
numpy       | Array library used for manipulating data
scipy       | Miscellaneous scientific functions
six         | Compatibility layer to support python 2 and 3


Optional python packages, which can be installed through `pip`:

Package     | Usage                                      
------------|-------------------------------------------
coverage    | For measuring test coverage
cma         | For optimizations using CMA-ES
numba       | For creating compiled code for faster processing
nose        | For parallel testing
simanneal   | Simulated annealing algorithm published on github


The classes in the project are organized as follows:
- We distinguish several different odor mixtures as inputs:
    - binary mixtures: ligands are either present or absent. The probability
        of ligands is controlled by an Ising-like distribution.
    - continuous mixtures: all ligands are present at random concentrations. The
        probability distribution of the concentration vector is specified by the
        mean concentrations and a covariance matrix.
    - sparse mixtures: the ligands that are present have random concentrations.
        Here, we use the algorithm from the binary mixtures to determine which
        ligands are present in a mixture and then chose their concentrations
        independently from exponential distributions. 
    The code for these mixtures is organized in different modules.
- The package `adaptive_response` is an extension of the `binary_response`,
    which looks into adaptive excitation thresholds.
- We distinguish between general classes and classes  with a concrete receptor
    library. Here, we distinguish libraries that do numerical simulations and
    libraries that provide analytical results.
