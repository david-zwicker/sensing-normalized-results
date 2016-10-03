'''
Created on Dec 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import logging
import unittest

import numpy as np
from scipy import misc

from .at_numeric import AdaptiveThresholdNumeric
from .at_theory import (AdaptiveThresholdTheory,
                        AdaptiveThresholdTheoryReceptorFactors)
from . import numba_patcher
from utils.testing import TestBase 

      

class TestLibraryPrimacyCoding(TestBase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    
    
    def _create_test_models(self, **kwargs):
        """ helper method for creating test models """
        # save numba patcher state
        numba_patcher_enabled = numba_patcher.enabled
        
        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['numba_enabled'] = (True, False)
        c_dists = AdaptiveThresholdNumeric.concentration_distributions
        settings['c_distribution'] = c_dists
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # try all these settings 
        for setting in setting_comb:
            # set the respective state of the numba patcher
            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = AdaptiveThresholdNumeric.create_test_instance(**kwargs)

            # create a meaningful error message for all cases
            model.settings = ', '.join("%s=%s" % v for v in setting.items())
            model.error_msg = ('The different implementations do not agree for '
                               + model.settings)
            yield model

        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)
        

    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = AdaptiveThresholdNumeric.create_test_instance()
        
        # probability of having m_s components in a mixture for h_i = h
        hval = np.random.random() - 0.5
        model.commonness = [hval] * model.Ns
        m_s = np.arange(0, model.Ns + 1)
        p_m = (misc.comb(model.Ns, m_s)
               * np.exp(hval*m_s)/(1 + np.exp(hval))**model.Ns)
        
        self.assertAllClose(p_m, model.mixture_size_distribution())
    
        # test random commonness and the associated distribution
        hs = np.random.random(size=model.Ns)
        model.commonness = hs
        self.assertAllClose(hs, model.commonness)
        model.substrate_probabilities = model.substrate_probabilities
        self.assertAllClose(hs, model.commonness)
        dist = model.mixture_size_distribution()
        self.assertAlmostEqual(dist.sum(), 1)
        ks = np.arange(0, model.Ns + 1)
        dist_mean = (ks*dist).sum()
        dist_var = (ks*ks*dist).sum() - dist_mean**2 
        stats = model.mixture_size_statistics() 
        self.assertAllClose((dist_mean, dist_var),
                            (stats['mean'], stats['var']))
        
        # probability of having m_s components in a mixture for h_i = h
        c_means = model.concentration_means
        for i, c_mean in enumerate(c_means):
            mean_calc = model.get_concentration_distribution(i).mean()
            pi = model.substrate_probabilities[i]
            self.assertAlmostEqual(c_mean, mean_calc * pi)
        
        # test setting the commonness
        commoness_schemes = [('const', {}),
                             ('single', {'p1': np.random.random()}),
                             ('single', {'p_ratio': 0.1 + np.random.random()}),
                             ('geometric', {'alpha': np.random.uniform(0.98, 1)}),
                             ('linear', {}),
                             ('random_uniform', {}),]
        
        for scheme, params in commoness_schemes:
            size1 = np.random.randint(1, model.Ns//2 + 1)
            size2 = np.random.randint(1, model.Ns//3 + 1) + model.Ns//2
            for mean_mixture_size in (size1, size2):
                model.choose_commonness(scheme, mean_mixture_size, **params)
                self.assertAllClose(model.mixture_size_statistics()['mean'],
                                    mean_mixture_size)


    def test_setting_threshold_factor(self):
        """ tests whether setting the coding_receptors is consistent """
        # construct specific model
        params = {'threshold_factor': .6}
        model = AdaptiveThresholdNumeric(6, 6, parameters=params)
        self.assertEqual(model.threshold_factor, .6)
        self.assertEqual(model.parameters['threshold_factor'], .6)
        
        # construct random model
        model = AdaptiveThresholdNumeric.create_test_instance(threshold_factor=.5)
        self.assertEqual(model.threshold_factor, .5)
        self.assertEqual(model.parameters['threshold_factor'], .5)
        
        # change coding_receptors
        model.threshold_factor = .4
        self.assertEqual(model.threshold_factor, .4)
        self.assertEqual(model.parameters['threshold_factor'], .4)
        
        
    def test_theory_consistency(self):
        """ do some consistency checks on the theory """
        theory0 = AdaptiveThresholdTheory.create_test_instance()

        # test backing out the threshold factor
        an = theory0.receptor_activity()
        alpha_est = theory0.threshold_factor_from_activity(an)
        self.assertAlmostEqual(alpha_est, theory0.threshold_factor, 5)
                
        d_max1 = theory0.activity_distance_uncorrelated(3)
        d_max2 = theory0.activity_distance_uncorrelated((3, 3))
        self.assertAlmostEqual(d_max1, d_max2)
        
        # test determining the threshold factor
        for n in (True, False):
            for i in (True, False):
                theory0.set_threshold_from_activity(0.1, n, i)
                alpha = theory0.threshold_factor
                theory0.set_threshold_from_activity_numeric(0.1, n, i)

                self.assertAlmostEqual(theory0.threshold_factor, alpha)
                self.assertAlmostEqual(theory0.receptor_activity(n, i), 0.1)
        
        
    def test_theory_receptor_factors_consistency(self):
        """ compares the two theory classes with each other """
        logging.disable(logging.WARN)
        
        theory0 = AdaptiveThresholdTheory.create_test_instance()
        theoryN = AdaptiveThresholdTheoryReceptorFactors(
            num_substrates=theory0.Ns, num_receptors=theory0.Nr,
            mean_sensitivity=theory0.mean_sensitivity, width=theory0.width,
            parameters=theory0.parameters.copy()
        )
        
        # test sensitivity statistics 
        stats0 = theory0.sensitivity_stats()
        statsN = theoryN.sensitivity_stats()
        self.assertEqual(stats0['mean'], statsN['mean'][0])
        self.assertEqual(stats0['var'], statsN['var'][0])
        
        # test excitation statistics
        stats0 = theory0.excitation_statistics()
        statsN = theoryN.excitation_statistics()
        self.assertEqual(stats0['mean'], statsN['mean'][0])
        self.assertEqual(stats0['var'], statsN['var'][0])
        
        # receptor activity
        an = theory0.receptor_activity(normalized_variables=False)
        self.assertAlmostEqual(an, theoryN.receptor_activity().mean())

        # mutual information
        MI = theory0.mutual_information(normalized_variables=False)
        self.assertAlmostEqual(MI, theoryN.mutual_information())


    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
        
    

if __name__ == '__main__':
    unittest.main()

