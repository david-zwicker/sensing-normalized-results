'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import itertools
import unittest

import numpy as np
from scipy import misc

from .lib_bin_base import LibraryBinaryBase
from .lib_bin_numeric import LibraryBinaryNumeric
from .numba_speedup import numba_patcher
from utils.testing import TestBase 

      

class TestLibraryBinary(TestBase):
    """ unit tests for the binary library """
    
    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    

    def _create_test_models(self):
        """ helper method for creating test models """
        # save numba patcher state
        numba_patcher_enabled = numba_patcher.enabled
        
        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['numba_enabled'] = (True, False)
        settings['mixture_correlated'] = (True, False)
        settings['fixed_mixture_size'] = (None, 2)
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # try all these settings 
        for setting in setting_comb:
            # set the respective state of the numba patcher
            numba_patcher.set_state(setting['numba_enabled'])
            
            # create test object
            model = LibraryBinaryNumeric.create_test_instance(
                        mixture_correlated=setting['mixture_correlated'],
                        fixed_mixture_size=setting['fixed_mixture_size']
                    )

            # create a meaningful error message for all cases
            model.error_msg = ('The different implementations do not agree for '
                               + ', '.join("%s=%s" % v for v in setting.items()))
            yield model

        # reset numba patcher state
        numba_patcher.set_state(numba_patcher_enabled)


    def test_base(self):
        """ consistency tests on the base class """
        # construct random model
        model = LibraryBinaryBase.create_test_instance()
        
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
                
                
    def test_mixture_entropy(self):
        """ test the calculations of the mixture entropy """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test the mixture entropy calculations
            self.assertAlmostEqual(model.mixture_entropy(),
                                   model.mixture_entropy_brute_force(),
                                   msg='Mixture entropy: ' + error_msg)
            
            self.assertAlmostEqual(model.mixture_entropy(),
                                   model.mixture_entropy_monte_carlo(),
                                   places=1,
                                   msg='Mixture entropy: ' + error_msg)


    def test_mixture_statistics(self):
        """ test mixture statistics calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check the mixture statistics
            c_stats_1 = model.mixture_statistics_brute_force()
            if not model.is_correlated_mixture:
                c_stats_2 = model.mixture_statistics()
                self.assertDictAllClose(c_stats_1, c_stats_2,
                                        rtol=5e-2, atol=5e-2,
                                        msg='Mixture statistics: ' + error_msg)

            c_stats_3 = model.mixture_statistics_monte_carlo()
            self.assertDictAllClose(c_stats_1, c_stats_3,
                                    rtol=5e-2, atol=5e-2,
                                    msg='Mixture statistics: ' + error_msg)
                
                
    def test_receptor_crosstalk(self):
        """ test receptor activity calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            fixed_mixture = model.parameters['fixed_mixture_size'] is not None
            if fixed_mixture and numba_patcher.enabled:
                self.assertRaises(NotImplementedError,
                                  model.receptor_crosstalk, "brute-force")
                
            else:
                q_nm_1 = model.receptor_crosstalk("brute-force")
                q_nm_2 = model.receptor_crosstalk("monte-carlo")
    
                self.assertAllClose(q_nm_1, q_nm_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor crosstalk: ' + error_msg)
    
                
    def test_receptor_activity(self):
        """ test receptor activity calculations """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            fixed_mixture = model.parameters['fixed_mixture_size'] is not None
            if fixed_mixture and numba_patcher.enabled:
                self.assertRaises(NotImplementedError,
                                  model.receptor_activity_brute_force)
                
            else:
                r_n_1, r_nm_1 = model.receptor_activity_brute_force(ret_correlations=True)
                r_n_2, r_nm_2 = model.receptor_activity_monte_carlo(ret_correlations=True)
    
                self.assertAllClose(r_n_1, r_n_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor activities: ' + error_msg)
                self.assertAllClose(r_nm_1, r_nm_2, rtol=5e-2, atol=5e-2,
                                    msg='Receptor correlations: ' + error_msg)
                
                
    def test_correlations_and_crosstalk(self):
        """ tests the correlations and crosstalk """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # check for known exception where the method are not implemented 
            fixed_mixture = model.parameters['fixed_mixture_size'] is not None
            if fixed_mixture and numba_patcher.enabled:
                self.assertRaises(NotImplementedError,
                                  model.receptor_activity_brute_force)
            
            else:
                for method in ('auto', 'estimate'):
                    if method == 'estimate':
                        kwargs = {'approx_prob': True}
                    else:
                        kwargs = {}
                    r_n, r_nm = model.receptor_activity(
                                        method, ret_correlations=True, **kwargs) 
                    q_n, q_nm = model.receptor_crosstalk(
                                   method, ret_receptor_activity=True, **kwargs)
                    
                    self.assertAllClose(r_n, q_n, rtol=5e-2, atol=5e-2,
                                        msg='Receptor activities: ' + error_msg)
                    #r_nm_calc = np.clip(np.outer(q_n, q_n) + q_nm, 0, 1)
                    r_nm_calc = np.outer(q_n, q_n) + q_nm
                    self.assertAllClose(r_nm, r_nm_calc, rtol=0, atol=0.5,
                                        msg='Receptor correlations: ' + error_msg)
                
                
    def test_mututal_information(self):
        """ test mutual information calculation """
        for model in self._create_test_models():
            error_msg = model.error_msg
            
            # test calculation of mutual information
            MI_1 = model.mutual_information_brute_force()
            MI_2 = model.mutual_information_monte_carlo()

            self.assertAllClose(MI_1, MI_2, rtol=5e-2, atol=5e-2,
                                msg='Mutual information: ' + error_msg)
    
    
    def test_numba_consistency(self):
        """ test the consistency of the numba functions """
        # this tests the numba consistency for uncorrelated mixtures
        self.assertTrue(numba_patcher.test_consistency(repeat=3, verbosity=1),
                        msg='Numba methods are not consistent')
    
    
    def test_numba_consistency_special(self):
        """ test the consistency of the numba functions """

        # collect all settings that we want to test
        settings = collections.OrderedDict()
        settings['mixture_correlated'] = (True, False)
        settings['fixed_mixture_size'] = (None, 2)
        
        # create all combinations of all settings
        setting_comb = [dict(zip(settings.keys(), items))
                        for items in itertools.product(*settings.values())]
        
        # define the numba methods that need to be tested
        numba_methods = ('LibraryBinaryNumeric.mutual_information_brute_force',
                         'LibraryBinaryNumeric.mutual_information_monte_carlo')
        
        # try all these settings 
        for setting in setting_comb:
            # create a meaningful error message for all cases
            error_msg = ('The Numba implementation is not consistent for ' +
                         ', '.join("%s=%s" % v for v in setting.items()))
            # test the number class
            for name in numba_methods:
                consistent = numba_patcher.test_function_consistency(
                                    name, repeat=2, instance_parameters=setting)
                if not consistent:
                    self.fail(msg=name + '\n' + error_msg)
    
    
    def test_optimization_consistency(self):
        """ test the various optimization methods for consistency """
        
        # list all the tests that should be done
        tests = [
            {'method': 'descent', 'multiprocessing': False},
            {'method': 'descent', 'multiprocessing': True},
            {'method': 'descent_multiple', 'multiprocessing': False},
            {'method': 'descent_multiple', 'multiprocessing': True},
            {'method': 'anneal'},
        ]
        
        # initialize a model
        model = LibraryBinaryNumeric.create_test_instance()
        
        MI_ref = None
        for test_parameters in tests:
            MI, _ = model.optimize_library('mutual_information', direction='max',
                                           steps=1e4, **test_parameters)
            
            if MI_ref is None:
                MI_ref = MI
            else:
                msg = ("Optimization inconsistent (%g != %g) for %s"
                       % (MI_ref, MI, str(test_parameters)))
                self.assertAllClose(MI, MI_ref, rtol=.1, atol=.1, msg=msg)
        
        

if __name__ == '__main__':
    unittest.main()
    

