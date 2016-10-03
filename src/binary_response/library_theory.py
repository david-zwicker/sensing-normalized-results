'''
Created on Sep 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np

from utils.math.distributions import lognorm_mean, DeterministicDistribution

__all__ = ['LibraryLogNormal']



class LibraryLogNormal(object):
    """ represents a single receptor library with random entries drawn from a
    log-normal distribution """


    def __init__(self, mixture, mean_sensitivity=1, correlation=0, **kwargs):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix.
        The width of the distribution is either set by the parameter `width` or
        by setting the `standard_deviation`.
        """
        self.mixture = mixture
        self.mean_sensitivity = mean_sensitivity
        self.correlation = correlation

        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            cv = standard_deviation / mean_sensitivity 
            self.width = np.sqrt(np.log(cv**2 + 1))
        elif 'width' in kwargs:
            self.width = kwargs.pop('width')
        else:
            standard_deviation = 1
            cv = standard_deviation / mean_sensitivity 
            self.width = np.sqrt(np.log(cv**2 + 1))

        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 
            
            
    @property
    def standard_deviation(self):
        """ return the standard deviation of the distribution """
        return self.mean_sensitivity * np.sqrt((np.exp(self.width**2) - 1))
            

    @property
    def sensitivity_distribution(self):
        """ returns the sensitivity distribution """
        if self.correlation != 0:
            raise NotImplementedError('Cannot return the sensitivity '
                                      'distribution with correlations, yet')
        
        if self.width == 0:
            return DeterministicDistribution(self.mean_sensitivity)
        else:
            return lognorm_mean(self.mean_sensitivity, self.width)


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        var = S0**2 * (np.exp(self.width**2) - 1)
        covar = S0**2 * (np.exp(self.correlation * self.width**2) - 1)
        return {'mean': S0, 'std': np.sqrt(var), 'var': var, 'cov': covar}


    def get_optimal_parameters(self, fixed_parameter='S0'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        if self.mixture.is_correlated_mixture:
            logging.warning('The optimization has not been tested for '
                            'correlated mixtures')

        ctot_stats = self.mixture.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']
        ctot_cv2 = ctot_var/ctot_mean**2
        
        if fixed_parameter == 'width':
            # keep the width parameter fixed and determine the others 
            width_opt = self.width
            
            arg = 1 + ctot_cv2 * np.exp(width_opt**2)
            S0_opt = np.sqrt(arg) / ctot_mean
            std_opt = S0_opt * np.sqrt(np.exp(width_opt**2) - 1)
            
        elif fixed_parameter == 'S0':
            # keep the typical sensitivity fixed and determine the other params 
            S0_opt = self.mean_sensitivity
            
            arg = (ctot_mean**2 * self.mean_sensitivity**2 - 1)/ctot_cv2
            if arg >= 1:
                width_opt = np.sqrt(np.log(arg))
                std_opt = self.mean_sensitivity * np.sqrt(arg - 1)
            else:
                logging.warning('Given mean sensitivity is too small to find a '
                                'suitable width parameter')
                width_opt = 0
                std_opt = 0
            
        else:
            raise ValueError('Parameter `%s` is unknown or cannot be held '
                             'fixed' % fixed_parameter) 
        
        return {'mean_sensitivity': S0_opt, 'width': width_opt,
                'standard_deviation': std_opt}
    
    
    def get_optimal_library(self, fixed_parameter='S0'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        library_opt = self.get_optimal_parameters(fixed_parameter)
        return {'distribution': 'log_normal', 'width': library_opt['width'],
                'mean_sensitivity': library_opt['mean_sensitivity'],
                'correlation': 0}
        
        
        