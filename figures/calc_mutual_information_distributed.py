#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import time
import pickle
from collections import OrderedDict

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from binary_response import *

from utils.math.stats import StatisticsAccumulator
from adaptive_response.adaptive_threshold import (AdaptiveThresholdTheory,
                                                  AdaptiveThresholdTheoryReceptorFactors)


Nr, alpha = 32, 1.5
Ns, s = 256, 32
s = 0.1 * Ns
#r_list = [2, 4, 8]
an_list = [0.5, 0.2, 0.1]
width = 1

# Nr, alpha = 8, 1.3
# Ns, s = 128, 32
# r = [2, 4]
#widths = [1, 2]


data = OrderedDict()

def get_alpha_from_an(alpha, theory, an):
    """ helper function """
    theory.threshold_factor = alpha
    return an - np.mean(theory.receptor_activity())


for an in an_list:
    print('an=%g' % an)
    theory = AdaptiveThresholdTheoryReceptorFactors(
        Ns, Nr,
        mean_sensitivity=1, width=width,
        parameters={'c_distribution': 'log-normal'})
    theory.threshold_factor = alpha
    theory.choose_commonness('const', mean_mixture_size=s)
    theory.c_means = 1
    theory.c_vars = 1

    variances = np.linspace(0, 1, 16)
    MI_mean = []
    MI_std = []
    an_list = []
    for variance in variances:

        MI_stats = StatisticsAccumulator()
        for _ in xrange(1000):
            theory.choose_receptor_factors('log_normal', variance=variance)

            # determine the threshold factor
            try:
                alpha = optimize.brentq(get_alpha_from_an, 0.1, 5, args=(theory, an))
            except ValueError:
                alpha = optimize.newton(get_alpha_from_an, 2, args=(theory, an))
            theory.threshold_factor = alpha

            MI_stats.add(theory.mutual_information(warn=False))

        MI_mean.append(MI_stats.mean)
        MI_std.append(MI_stats.std)

        ans = theory.receptor_activity()
        an_list.append(ans)

    data[an] = {'MI_mean': np.array(MI_mean),
                'MI_std': np.array(MI_std)}


res = {'variances': variances,
       'data': data}

with open('data/mutual_information_distributed.pkl', 'wb') as fp:
    pickle.dump(res, fp)
