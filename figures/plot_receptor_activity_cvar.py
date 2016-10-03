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
Ns, s = 128, 32
width = 1
alphas = [1.5, 2, 3]

parameters = {'c_distribution': 'log-normal',
              'ctot_method': 'leastsq',
              'compensated_threshold': False}


c_vars = np.logspace(-2.3, 3.3, 128)
c_stds = np.sqrt(c_vars)


theories = {}
for alpha in alphas:
    continue
    theory = AdaptiveThresholdTheory(Ns, Nr, width=width, parameters=parameters)
    theory.threshold_factor = alpha
    theory.choose_commonness('const', mean_mixture_size=s)
    theory.c_means = 1

    ans = []
    for c_var in c_vars:
        theory.c_vars = c_var
        an_th = theory.receptor_activity(normalized_variables=True, integrate=False)
        ans.append(an_th)

    theories[alpha] = ans


colors = [cm.plasma(x) for x in np.linspace(0, 0.9, len(alphas))]


for fig in figures(
        'receptor_activity_cvar.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    for alpha, color in zip(alphas, colors):
        # load data
        data = pd.DataFrame.from_csv('data/receptor_activity_cvar_alpha_%g.csv' % alpha)

        plt.plot(np.sqrt(data['c_var']), data['an_num_mean'], #yerr=data['an_num_std'],
                 '-', ms=3, color=color, label=r'$\alpha = %g$' % alpha)

#         plt.errorbar(np.sqrt(data['c_var']), data['an_num_mean'], yerr=data['an_num_std'],
#                      color=color, label=r'$\alpha = %g$' % alpha)

        continue
        c_hat_mean = data['c_hat_mean']
        c_hat_var = data['c_hat_var']
        S_var = theory.sensitivity_stats()['var']
        e_hat_var = S_var * Ns * (c_hat_mean**2 + c_hat_var)

        res = [lognorm_mean_var(1, v).sf(alpha)
               for v in np.array(e_hat_var)]

#         plt.plot(np.sqrt(data['c_var']), res, color=color, lw=1,
#                  label=r'$\alpha = %g$' % alpha)


        e_hat_var = S_var * (1 + c_vars) / s
        res = [lognorm_mean_var(1, v).sf(alpha)
               for v in np.array(e_hat_var)]
        plt.plot(c_stds, res, color=color, lw=1,
                 label=r'$\alpha = %g$' % alpha)


        #plt.plot(c_vars, theories[alpha], ':', color=color, lw=.5)


    plt.axhline(1/300, ls=':', color='0.5')

#    plt.legend(loc='lower right', bbox_to_anchor=(0.02, 0.05), fontsize=8)
    plt.legend(loc='lower right', fontsize=7)
    plt.xscale('log')
    plt.yscale('log')
#    plt.ylim(5e-3, 2)
    plt.xlim(c_stds.min(), c_stds.max())
    plt.ylim(1e-5, 2)

    plt.xlabel(r'Concentration variability $\sigma/\mu$')
    plt.ylabel(r'Activity $\mean{a_n}$')

