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


Nr, alpha = 32, 1.5
Ns = 256
s = 0.1 * Ns
#r_list = [6, 4, 2]
an_list = [0.5, 0.2, 0.1]
width = 1


with open('data/mutual_information_an.pkl', 'rb') as fp:
    res = pickle.load(fp)

factors = res['factors']
df = pd.DataFrame(res['data'])
df.set_index(['an', 'width'], inplace=True)

colors = [cm.viridis(x) for x in np.linspace(0, 0.9, len(an_list))]

for fig in figures(
        'mutual_information_an.pdf',
        fig_width_pt=200., crop_pdf=False, legend_frame=False,
        transparent=True, post_process=True,
#        num_ticks=3
    ):

    plt.axhline(0, ls=':', color='k')

    for k, an in enumerate(an_list):
        thresh = df.ix[an].ix[width]['MI_less']
        MI_rel = df.ix[an].ix[width]['MI']
        plt.plot(factors, (MI_rel - thresh) , '-',
                 label=r'$\mean{a_n}=%g$' % an, color=colors[k])

        #max_id = np.argmax(MI_rel)
        #idx = np.flatnonzero(MI_rel[max_id:] < thresh) + max_id
        #print('xi_1 max = %g for width = %g' % (factors[idx[0]], width))

    plt.legend(loc='lower left', fontsize=8)
#    plt.yscale('log')
    plt.xlim(0, 2.45)
    plt.ylim(-1.2, 1.2)

    xs = np.arange(0, 1.7, .5)
    plt.xticks(xs, [r'$%g$' % x for x in xs])

    #ys = np.arange(0, .7, .2)
    #plt.yticks(ys, [r'$\unit[%g]{\%%}$' % y for y in ys])

    #plt.xlabel(r'Receptor sensitivity $\langle S_{n1} \rangle$')#\gamma_1$')
    plt.xlabel(r'Sensitivity $\xi_1$ of receptor 1')
    plt.ylabel(r'$I-I_0$ $[\unit{bits}]$')
