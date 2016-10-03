#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import multiprocessing as mp
import itertools

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
Ns = 8*2048
width = 1
alphas = [1.5, 2, 3]
cvars = [1, 100] # [0.1, 1, 10]

calc_concentration_statistics = False
calc_theory = False


sizes = np.unique(logspace(1, 3e3, 32).astype(np.int))

parameters = {'c_distribution': 'log-normal',
              'ensemble_average_num': 8}


def calculate((alpha, cvar)):
    print('Handle alpha=%g, cvar=%g' % (alpha, cvar))

    model = AdaptiveThresholdNumeric(Ns, Nr, parameters=parameters)
    model.threshold_factor = alpha
    model.c_means = 1
    model.c_vars = cvar
    model.choose_sensitivity_matrix('log-normal', mean_sensitivity=1, width=width)

    init_state = model.parameters['initialize_state']
    init_state['c_mean'] = 'exact'
    init_state['c_var'] = 'exact'
    init_state['correlations'] = 'exact'

    theory = AdaptiveThresholdTheory.from_other(model, mean_sensitivity=1, width=width)

    en_stats = theory.excitation_statistics(normalized=True)
    print('Theory excitation statistics: mean=%g, var=%g' %
           (en_stats['mean'], en_stats['var']))


    data = []
    for s in sizes:
        print('Handle s = %g' % s)
        model.choose_commonness('const', mean_mixture_size=s)
        theory.choose_commonness('const', mean_mixture_size=s)

        ans = model.receptor_activity()

        res = {'s': s,
               'Nr': model.Nr,
               'an_num_mean': ans.mean(),
               'an_num_std': ans.std()}

        if calc_concentration_statistics:
            c_hat_stats = StatisticsAccumulator()
            for ci in model._sample_mixtures():
                ctot = ci.sum()
                if ctot > 0:
                    ci /= ctot
                c_hat_stats.add(ci)

            res['c_hat_mean'] = c_hat_stats.mean.mean()
            res['c_hat_var'] = c_hat_stats.var.mean()

        if calc_theory:
            theory.parameters['compensated_threshold'] = True

            an_th = theory.receptor_activity(normalized_variables=False, integrate=False)
            an_th_norm = theory.receptor_activity(normalized_variables=True, integrate=False)

            res['an_th_comp'] = an_th,
            res['an_th_norm_comp'] = an_th_norm

            theory.parameters['compensated_threshold'] = False

            an_th = theory.receptor_activity(normalized_variables=False, integrate=False)
            an_th_norm = theory.receptor_activity(normalized_variables=True, integrate=False)

            res['an_th'] = an_th,
            res['an_th_norm'] = an_th_norm

        data.append(res)

    data = pd.DataFrame(data)
    data.to_csv('data/receptor_activity_s_alpha_%g_cvar_%g.csv' % (alpha, cvar))


mp.Pool().map(calculate, itertools.product(alphas, cvars))
