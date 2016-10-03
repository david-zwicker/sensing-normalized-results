#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import multiprocessing as mp

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
s = 32
s = 0.1 * Ns

width = 1
alphas = [1, 1.5, 2, 3]
alphas = [1.5, 2, 3]
c_vars = np.logspace(-2.3, 3.3, 16)

calc_concentration_statistics = False
calc_theory = False

parameters = {'c_distribution': 'log-normal',
              'ensemble_average_num': 8}


def calculate(alpha):
    print('Handle alpha = %g' % alpha)

    model = AdaptiveThresholdNumeric(Ns, Nr, parameters=parameters)
    model.threshold_factor = alpha
    model.choose_commonness('const', mean_mixture_size=s)
    model.c_means = 1
    model.c_vars = 1
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
    for c_var in c_vars:
        print('Handle c_var = %g' % c_var)
        model.c_vars = c_var

        ans_mean, ans_std = model.ensemble_average('receptor_activity',
                                                   avg_num=128, multiprocessing=True)

        res = {'c_var': c_var,
               'Nr': model.Nr,
               'an_num_mean': ans_mean.mean(),
               'an_num_std': ans_std.mean()}

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
            theory.c_vars = c_var
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
    data.to_csv('data/receptor_activity_cvar_alpha_%g.csv' % alpha)


map(calculate, alphas)
