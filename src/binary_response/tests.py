'''
Created on May 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

import numpy as np

from .library_base import LibraryBase
from utils.testing import TestBase

      
      
class TestLibraryBase(TestBase):
    """ unit tests for the continuous library """

    _multiprocess_can_split_ = True #< let nose know that tests can run parallel
    

    def test_base_class(self):
        """ test the base class """
        obj = LibraryBase.create_test_instance()
        
        # calculate mutual information
        for method in ('expansion', 'hybrid', 'polynom'):
            q_n = np.full(obj.Nr, 0.1) + 0.8*np.random.rand()
            q_nm = np.full((obj.Nr, obj.Nr), 0.1) + 0.1*np.random.rand()

            np.fill_diagonal(q_nm, 0)
            q_nm_mean = q_nm[~np.eye(obj.Nr, dtype=np.bool)].mean()
            q_nm_var = q_nm[~np.eye(obj.Nr, dtype=np.bool)].var()
            
            MI1 = obj._estimate_MI_from_q_values(q_n, q_nm, method=method)
            MI2 = obj._estimate_MI_from_q_stats(
                q_n.mean(), q_nm_mean, q_n.var(), q_nm_var,
                method=method
            )
            msg = 'Mutual informations do not agree for method=`%s`' % method
            self.assertAllClose(MI1, MI2, rtol=0.1, msg=msg)
                    
    

if __name__ == '__main__':
    unittest.main()
