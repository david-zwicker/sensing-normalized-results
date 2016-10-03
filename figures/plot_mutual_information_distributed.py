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
from adaptive_response.adaptive_threshold import AdaptiveThresholdTheoryReceptorFactors


Nr, alpha = 16, 1.5
Ns, s = 128, 32
#r_list = [8, 4, 2]
an_list = [0.5, 0.2, 0.1]


with open('data/mutual_information_distributed.pkl', 'rb') as fp:
    res = pickle.load(fp)

variances = res['variances']
data = res['data']

colors = [cm.viridis(x) for x in np.linspace(0, 0.9, len(an_list))]

for fig in figures(
        'mutual_information_distributed.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, #post_process=False,
#        num_ticks=3
    ):

    #thresh = data[widths[0]]['MI_less'] / Na
    #plt.axhline(thresh, ls=':', color=COLOR_RED)


    for k, an in enumerate(an_list):
        errorplot(variances, data[an]['MI_mean'], yerr=data[an]['MI_std'],
                  label=r'$\mean{a_n}=%g$' % an, color=colors[k])

#         max_id = np.argmax(MI_rel)
#         idx = np.flatnonzero(MI_rel[max_id:] < thresh) + max_id
#         print('xi_1 max = %g for width = %g' % (factors[idx[0]], width))

    plt.legend(loc='best', fontsize=8)
#    plt.yscale('log')
    plt.xlim(0, variances.max())
    plt.ylim(0, 34)

    #plt.xlabel(r'Receptor sensitivity $\langle S_{n1} \rangle$')#\gamma_1$')
    plt.xlabel(r'Sensitivity variation $\var(\xi_n)/\mean{\xi_n}^2$')
    plt.ylabel(r'Infor. $I$ [$\unit{bits}$]')
