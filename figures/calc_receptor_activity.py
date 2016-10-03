#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import numpy as np
from scipy import special, optimize, stats
import matplotlib.pyplot as plt
import numba
import pandas as pd

from plotting_functions import *

from binary_response.sparse_mixtures import LibrarySparseNumeric
from adaptive_response import AdaptiveThresholdNumeric, AdaptiveThresholdTheory
from utils.math.distributions import lognorm_mean_var
from utils.math.stats import StatisticsAccumulator


Nr = 32
Ns = 256
s = 0.1 * Ns
width = 1
c_vars = [1e0]


for cvar in c_vars:
    parameters = {'c_distribution': 'log-normal',
                  'ensemble_average_num': 8}

    model = AdaptiveThresholdNumeric(Ns, Nr, parameters=parameters)
#    model.threshold_factor = alpha
    model.choose_commonness('const', mean_mixture_size=s)
    model.c_means = 1
    model.c_vars = cvar
    model.choose_sensitivity_matrix('log-normal', mean_sensitivity=1, width=width)


    init_state = model.parameters['initialize_state']
    init_state['c_mean'] = 'exact'
    init_state['c_var'] = 'exact'
    init_state['correlations'] = 'exact'

    theories = [AdaptiveThresholdTheory.from_other(model, mean_sensitivity=1, width=f*width)
                for f in [0.5, 1, 2]]


    # en_stats = model.excitation_statistics(normalized=False)
    # print('Numerical excitation statistics (not normalized): mean=%g, var=%g' %
    #         (en_stats['mean'].mean(),
    #          en_stats['var'].mean()))
    #
    # en_stats = model.excitation_statistics(normalized=True)
    # print('Numerical excitation statistics (normalized): mean=%g, var=%g' %
    #         (en_stats['mean'].mean(),
    #          en_stats['var'].mean()))

    en_stats = theories[1].excitation_statistics(normalized=True)
    print('Theory excitation statistics: mean=%g, var=%g' %
            (en_stats['mean'], en_stats['var']))


    factors = np.linspace(0, 2.5, 16)
    data = []
    for factor in factors:
        model.threshold_factor = factor

        ans = model.receptor_activity()
        MI_num = model.mutual_information('moments')

        res = {'factor': factor,
                'Nr': model.Nr,
                'an_num_mean': ans.mean(),
                'an_num_std': ans.std(),
                'MI_num': MI_num}

        for k, theory in enumerate(theories):
            theory.threshold_factor = factor
            theory.parameters['compensated_threshold'] = True

            an_th = theory.receptor_activity(normalized_variables=False, integrate=False)
            an_th_norm = theory.receptor_activity(normalized_variables=True, integrate=False)

            res['an_th_comp_%d' % k] = an_th,
            res['an_th_norm_comp_%d' % k] = an_th_norm

            theory.parameters['compensated_threshold'] = False

            an_th = theory.receptor_activity(normalized_variables=False, integrate=False)
            an_th_norm = theory.receptor_activity(normalized_variables=True, integrate=False)

            res['an_th_%d' % k] = an_th,
            res['an_th_norm_%d' % k] = an_th_norm

        data.append(res)

    data = pd.DataFrame(data)
    data.to_csv('data/receptor_activity_cvar_%g.csv' % np.log10(cvar))
