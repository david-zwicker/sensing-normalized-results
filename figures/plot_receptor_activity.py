#!/usr/bin/env python2

from __future__ import division

import sys, os.path
sys.path.append(os.path.expanduser('~/Code/sensing'))

import time
import pickle

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd

from binary_response import *


from figure_presets import *
from plotting_functions import *


Ns = 128
c_vars = [1e2, 1e0, 1e-2]


# load data
data = pd.DataFrame.from_csv('data/receptor_activity_cvar_0.csv')



for fig in figures(
        'receptor_activity.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    plt.plot(data['factor'], data['an_num_mean'], #yerr=data['an_num_std'],
             'o', color=COLOR_ORANGE, label=r'Numerics')
#    plt.plot(data['factor'], data['an_th'], 'b-', label='Theory')
    plt.plot(data['factor'], data['an_th_norm_1'], color='k', lw=1.5,
             label=r'Approximation, Eq. \bf{6}')
    #plt.plot(data['factor'], data['an_th_norm_comp_1'], ':', color=COLOR_ORANGE, label='')

    plt.axhline(1/300, ls=':', color='0.5')

#    plt.plot(data['factor'], data['an_th_norm_0'], ':', color=COLOR_ORANGE, label='')

    plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.05), fontsize=8)
    plt.yscale('log')
#    plt.ylim(5e-3, 2)
    plt.xlim(0, 2.6)
    plt.ylim(2.5e-3, 2)

    plt.xlabel(r'Inhibition strength $\alpha$')
    plt.ylabel(r'Activity $\mean{a_n}$')

