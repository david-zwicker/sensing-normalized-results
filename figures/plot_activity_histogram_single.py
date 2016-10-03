#!/usr/bin/env python2

from __future__ import division

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../src'))

import time
import pickle

import numpy as np
from scipy import optimize, stats
import matplotlib.pyplot as plt
import pandas as pd

from binary_response import *

Nr = 32

from figure_presets import *
from plotting_functions import *

alphas = [1.4]

for alpha in alphas:
    print('alpha = %g' % alpha)
    # load data
    data = pd.DataFrame.from_csv('data/activity_histogram_alpha_%g.csv' % alpha)
    Na_mean_est = data['mean_est'][0]


    with figure_file(
            'activity_histogram_single_alpha_%g.pdf' % alpha,
            fig_width_pt=200., crop_pdf=False, legend_frame=False,
            transparent=True, #post_process=False,
    #        num_ticks=3
        ) as fig:

        w = 0.4
        data_norm = data['count'] / data['count'].sum()
        plt.bar(data['Na'] - w, data_norm, width=2*w,
                color=COLOR_ORANGE)

        Na_mean = (data['Na'] * data['count']).sum() / data['count'].sum()

        #an_mean = Na_mean_est / Nr
        an_mean = Na_mean / Nr
        Na_std = np.sqrt(Nr * an_mean * (1 - an_mean))

        dist = stats.binom(Nr, an_mean)
        x = np.array(data['Na'])
        y = dist.pmf(x)

        plt.plot(x, y, '-o', color='k', ms=3)

        plt.xlim(-0.5, np.round(Na_mean_est + 2*Na_std) + .5)
        plt.ylim(0, 1.1 * data_norm.max())

        plt.xlabel(r'Active channel count $\sum_n a_n$')
        plt.ylabel('Frequency')

