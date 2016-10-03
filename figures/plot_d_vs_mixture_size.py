#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import time
import pickle
from collections import OrderedDict

import numpy as np
from scipy import optimize, special
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from binary_response import *

from figure_presets import *
from plotting_functions import *
from adaptive_response.adaptive_threshold import AdaptiveThresholdTheory


Nr = 64
width = 1

alphas = [1.5, 2, 3]
#widths = np.linspace(0, 3.2, 64)[1:]#[0.1, 1, 2]
sizes = np.arange(1, 25)
colors = [cm.plasma(x) for x in np.linspace(0, 0.9, len(alphas))]


data = []
for alpha in alphas:
    print 'Handle alpha =', alpha
    res = []
    for size in sizes:
        theory = AdaptiveThresholdTheory(2, Nr, mean_sensitivity=1, width=width,
                                         parameters={'c_distribution': 'log-normal'})
        theory.threshold_factor = alpha
        res.append(theory.activity_distance_target_background(c_ratio=1/size, background_size=size))
    data.append(res)

data = np.array(data)


for fig in figures(
        'd_vs_mixture_size.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    plt.axhline(2/50, xmax=0.6, color='0.5', ls=':')
    plt.axhline(2/300, color='0.5', ls=':')
    plt.axhline(2/1000, color='0.5', ls=':')

    for k, alpha in enumerate(alphas):
        plt.plot(sizes, data[k] / Nr, '-', ms=3, color=colors[k], label=r'$\alpha = %g$' % alpha)


#     varE = np.exp(widths**2) - 1
#     zeta = 0.5*np.log(1 + varE)
#     alpha = np.exp(zeta)
#     arg = (np.log(alpha) + zeta)/(2*np.sqrt(zeta))
#     an_est = 0.5*special.erfc(arg)
#     d = 2 * an_est * (1 - an_est)
#     plt.plot(widths, d, '--', color='0.75', label='')

#        width_opt = np.sqrt(np.log(alpha**2))
#        plt.axvline(width_opt, color=colors[k])

    plt.legend(loc='best', fontsize=8)
    #plt.ylim(2e-4, 0.3)

    plt.xscale('log')
    plt.xlim(1, sizes.max())

    plt.yscale('log')
    plt.ylim(5e-4, 0.3)

#    plt.xticks((1, 10, 20, 30))

    #plt.xlabel(r'Receptor sensitivity $\langle S_{n1} \rangle$')#\gamma_1$')
    plt.xlabel(r'Background mixture size $s$')
    plt.ylabel(r'Dist. $\mean{d}/N_{\rm R}$')

