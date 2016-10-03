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


Nr = 32
Ns = 256
s = 0.1 * Ns
width = 1

alphas = [1.4]

for alpha in alphas:
    print('alpha = %g' % alpha)

    parameters = {'c_distribution': 'log-normal',
                  'ensemble_average_num': 8}

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

    Nc_mean_est = theory.receptor_activity() * model.Nr
    Nc_mean_est

    an_count = np.zeros(model.Nr + 1)
    for _ in xrange(50):
        model.choose_sensitivity_matrix('log-normal', mean_sensitivity=1, width=width)
        for a_n in model._sample_activities(1000):
            an_count[a_n.sum()] += 1

    data = [{'Na': k, 'count': count, 'mean_est': Nc_mean_est}
            for k, count in enumerate(an_count)]

    data = pd.DataFrame(data)
    data.to_csv('data/activity_histogram_alpha_%g.csv' % alpha)
