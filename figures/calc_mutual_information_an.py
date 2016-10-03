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

from figure_presets import *
from plotting_functions import *
from adaptive_response.adaptive_threshold import (AdaptiveThresholdTheory,
                                                  AdaptiveThresholdTheoryReceptorFactors)


Nr, alpha = 32, 1.5
Ns = 256
s = 0.1 * Ns
#r_list = [2, 4, 6, 8
an_list = [0.5, 0.2, 0.1, 0.05, 0.02]
widths = [1, 2]


factors = np.linspace(0, 3.1, 128)


data = []

def get_alpha_from_an(alpha, theory, an):
    """ helper function """
    theory.threshold_factor = alpha
    return an - np.mean(theory.receptor_activity())


for an in an_list:
    for width in widths:
        print('an=%g, width=%g' % (an, width))

        theory = AdaptiveThresholdTheoryReceptorFactors(
            Ns, Nr,
            mean_sensitivity=1, width=width,
            parameters={'c_distribution': 'log-normal'})
        theory.threshold_factor = alpha
        theory.choose_commonness('const', mean_mixture_size=s)
        theory.c_means = 1
        theory.c_vars = 1

        theory_less = AdaptiveThresholdTheory(
            Ns, Nr - 1,
            mean_sensitivity=1, width=width,
            parameters={'c_distribution': 'log-normal'})
        theory_less.threshold_factor = alpha
        theory_less.choose_commonness('const', mean_mixture_size=s)
        theory_less.c_means = 1
        theory_less.c_vars = 1

        alpha = theory_less.set_threshold_from_activity(an * Nr / (Nr - 1))
        MI_less = theory_less.mutual_information(warn=False)

        MIs = []
        an_list = []
        for factor in factors:
            theory.receptor_factors = np.r_[factor, np.ones(Nr - 1)]

            # determine the threshold factor
            alpha = optimize.newton(get_alpha_from_an, alpha, args=(theory, an))
            theory.threshold_factor = alpha
            MIs.append(theory.mutual_information(warn=False))

        data.append({'width': width,
                     'an': an,
                     'MI': np.array(MIs),
                     'MI_less': MI_less})


res = {'factors': factors,
       'data': data}

with open('data/mutual_information_an.pkl', 'wb') as fp:
    pickle.dump(res, fp)
