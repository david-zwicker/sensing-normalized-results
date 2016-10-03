#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import time
import pickle

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from adaptive_response import AdaptiveThresholdNumeric, AdaptiveThresholdTheory
from utils.math.distributions import lognorm_mean_var


from figure_presets import *
from plotting_functions import *


Nr = 32
Ns = 2048
width = 1
alphas = [1.5, 2, 3]
sizes = np.unique(np.logspace(0, 3.2, 64).astype(np.int))

colors = [cm.plasma(x) for x in np.linspace(0, 0.9, len(alphas))]


for fig in figures(
        'receptor_activity_s.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    for alpha, color in zip(alphas, colors):

        # load data
        data = pd.DataFrame.from_csv('data/receptor_activity_s_alpha_%g_cvar_1.csv' % alpha)
        plt.plot(data['s'], data['an_num_mean'], #yerr=data['an_num_std'],
                 '-', ms=3, color=color, label=r'$\alpha = %g$' % alpha)

        data = pd.DataFrame.from_csv('data/receptor_activity_s_alpha_%g_cvar_100.csv' % alpha)
        plt.plot(data['s'], data['an_num_mean'], #yerr=data['an_num_std'],
                 '--', ms=3, color=color, label=r'')

#         data = pd.DataFrame.from_csv('results_alpha_%g_cvar_0.01.csv' % alpha)
#         plt.plot(data['s'], data['an_num_mean'], #yerr=data['an_num_std'],
#                  ':', ms=3, color=color, label=r'')

#         data = pd.DataFrame.from_csv('results_alpha_%g_cvar_100.csv' % alpha)
#         plt.plot(data['s'], data['an_num_mean'], #yerr=data['an_num_std'],
#                  ':', ms=3, color=color, label='')

#         data = pd.DataFrame.from_csv('results_alpha_%g_cvar_0.01.csv' % alpha)
#         plt.plot(data['s'], data['an_num_mean'], #yerr=data['an_num_std'],
#                  ':', ms=3, color=color, label='')

    plt.axhline(1/300, ls=':', color='0.5')

    plt.legend(loc='lower left', fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
#    plt.ylim(5e-3, 2)
    plt.xlim(sizes.min(), 3e3)#sizes.max())
    plt.ylim(1e-5, 2)

    plt.xlabel(r'Mixture size $s$')
    plt.ylabel(r'Activity $\mean{a_n}$')

