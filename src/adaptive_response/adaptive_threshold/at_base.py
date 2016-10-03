'''
Created on Feb 22, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np



class AdaptiveThresholdMixin(object):
    """ mixin that adds code to allow setting and reading an excitation
    threshold for adaptive receptors """ 

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'threshold_factor': 1, 
    }
    

    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(AdaptiveThresholdMixin, self).repr_params
        params.insert(2, 'alpha=%g' % self.threshold_factor)
        return params
    
    
    @classmethod
    def get_random_arguments(cls, threshold_factor=None, **kwargs):
        """ create random args for creating test instances """
        args = super(AdaptiveThresholdMixin, cls).get_random_arguments(**kwargs)
        
        if threshold_factor is None:
            threshold_factor = 0.5 * np.random.random()
            
        args['parameters']['threshold_factor'] = threshold_factor

        return args
    
    
    @property
    def threshold_factor(self):
        """ return the number of receptors used for coding """
        return self.parameters['threshold_factor']
    
    
    @threshold_factor.setter
    def threshold_factor(self, alpha):
        """ set the number of receptors used for coding """
        self.parameters['threshold_factor'] = alpha
        
        
