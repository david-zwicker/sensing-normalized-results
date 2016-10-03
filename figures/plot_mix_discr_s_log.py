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
from adaptive_response.adaptive_threshold import AdaptiveThresholdTheory


Nr = 64
width = 1

theory = AdaptiveThresholdTheory(2, Nr, mean_sensitivity=1, width=width,
                                 parameters={'c_distribution': 'log-normal'})

alphas = [1.5, 2, 3]
colors = [cm.plasma(x) for x in np.linspace(0, 0.9, len(alphas))]

s_vals = [8, 16]
styles = ['-', '--']


with figure_file(
        'mix_discr_s_log.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
    ) as fig:

    plt.axhline(2/50, color='0.5', ls=':')
    plt.axhline(2/300, color='0.5', ls=':')
    plt.axhline(2/1000, xmin=0.4, color='0.5', ls=':')

    for alpha, color in zip(alphas, colors):
        theory.threshold_factor = alpha
        for s, ls in zip(s_vals, styles):
            sBs = np.arange(0, s)

            y = [theory.activity_distance_mixtures(s, sB) / Nr
                 for sB in sBs]

            plt.plot(sBs / s, y, ls, ms=3, color=color)

    for s, ls in zip(s_vals, styles):
        plt.plot([], [], ls, ms=3, color='k', label=r'$s = %g$' % s)

    plt.legend(loc='lower left', fontsize=8, handlelength=3)

#    plt.xscale('log')
    plt.xlim(0, 0.9)

    plt.yscale('log')
    plt.ylim(5e-4, 0.3)
    #plt.ylim(0, 0.22)

    #plt.xlabel(r'Receptor sensitivity $\langle S_{n1} \rangle$')#\gamma_1$')
    plt.xlabel(r'Composition similarity $s_{\rm B}/s$')
    plt.ylabel(r'Dist. $\mean{d}/N_{\rm R}$')

