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
from math_functions import logspace
from adaptive_response.adaptive_threshold import AdaptiveThresholdTheory


Nr = 64
width = 1

cs = logspace(4e-3, 3, 64)

alphas = [1.5, 2, 3]
colors = [cm.plasma(x) for x in np.linspace(0, 0.9, len(alphas))]


data = []
for k, alpha in enumerate(alphas):
    print 'Handle alpha =', alpha
    theory = AdaptiveThresholdTheory(2, Nr, mean_sensitivity=1, width=width,
                                     parameters={'c_distribution': 'log-normal'})
    theory.threshold_factor = alpha
    hs_th = [theory.activity_distance_target_background(c) for c in cs]
    data.append(hs_th)

data = np.array(data)


for fig in figures(
        'conc_res_alpha_loglog.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    plt.axhline(2/50, xmax=0.6, color='0.5', ls=':')
    plt.axhline(2/300, color='0.5', ls=':')
    plt.axhline(2/1000, color='0.5', ls=':')

    for k, alpha in enumerate(alphas):
        plt.plot(1/cs, data[k] / Nr, color=colors[k], label=r'$\alpha = %g$' % alpha)

    log_slope_indicator(25, 130, ymax=0.01, exponent=-1)

    plt.xscale('log')
    plt.yscale('log')

    plt.ylim(5e-4, 0.3)
    plt.xlim(xmin=1/cs.max(), xmax=1/cs.min())

    plt.legend(loc='upper right', fontsize=8)

    plt.xlabel(r'Concentration ratio $c_{\rm b} / c_{\rm t}$')
    plt.ylabel(r'Dist. $\mean{d} / N_{\rm R}$')

