'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import numpy as np
from scipy import optimize

from .lib_bin_base import LibraryBinaryBase


LN2 = np.log(2)



class LibraryBinaryUniform(LibraryBinaryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries. """


    def __init__(self, num_substrates, num_receptors, density=1,
                 parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the fraction `density`
        of substrates a single receptor responds to """
        super(LibraryBinaryUniform, self).__init__(num_substrates,
                                                   num_receptors, parameters)
        self.density = density


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryBinaryUniform, self).repr_params
        params.append('xi=%g' % self.density)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibraryBinaryUniform, self).init_arguments
        args['density'] = self.density
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryUniform, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        return args

    
    def receptor_activity(self, ret_correlations=False, approx_prob=False,
                          clip=True):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                            approx_prob=approx_prob)
        
        r_n = q_n
        r_nm = q_n**2 + q_nm
        
        if clip:
            r_n = np.clip(r_n, 0, 1)
            r_nm = np.clip(r_nm, 0, 1)
        
        if ret_correlations:
            return r_n, r_nm
        else:
            return r_n
        
        
    def receptor_crosstalk(self, ret_receptor_activity=False, approx_prob=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """

        p_i = self.substrate_probabilities
        
        # get probability q_n and q_nm that receptors are activated 
        if approx_prob:
            # use approximate formulas for calculating the probabilities
            q_n = self.density * p_i.sum()
            q_nm = self.density**2 * p_i.sum()
            
            # clip the result to [0, 1]
            q_n = np.clip(q_n, 0, 1)
            q_nm = np.clip(q_nm, 0, 1)

        else:
            # use better formulas for calculating the probabilities
            xi = self.density 
            q_n = 1 - np.prod(1 - xi * p_i)
            q_nm = np.prod(1 - (2*xi - xi**2) * p_i) - np.prod(1 - xi * p_i)**2
                
        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm

        
    def mutual_information(self,  approx_prob=False,
                           excitation_method='logarithm',
                           mutual_information_method='default',
                           clip=True):
        """ return a theoretical estimate of the mutual information between
        input and output.
            `excitation_method` determines which method is used to approximate
                the mutual information. Possible values are `logarithm`, and
                `overlap`, in increasing order of accuracy.
            `approx_prob` determines whether a linear approximation should be
                used to calculate the probabilities that receptors are active
        """
        if excitation_method =='logarithm':
            # use the expansion of the mutual information around the optimal
            # point to calculate an approximation of the mutual information
            
            # determine the probabilities of receptor activations        
            q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                                approx_prob=approx_prob)
    
            # calculate mutual information from this
            MI = self._estimate_MI_from_q_stats(
                                    q_n, q_nm, method=mutual_information_method)

        elif excitation_method == 'overlap':
            # calculate the MI assuming that receptors are independent.
            # This expression assumes that each receptor provides a fractional 
            # information H_r/N_s. Some of the information will be overlapping
            # and the resulting MI is thus smaller than the naive estimate:
            #     MI < N_r * H_r

            # determine the probabilities of receptor activation  
            q_n = self.receptor_activity(approx_prob=approx_prob)
    
            # calculate mutual information from this, ignoring crosstalk
            MI = self._estimate_MI_from_q_stats(
                                       q_n, 0, method=mutual_information_method)

            # estimate the effect of crosstalk by calculating the expected
            # overlap between independent receptors  
            H_r = MI / self.Nr
            MI = self.Ns - self.Ns*(1 - H_r/self.Ns)**self.Nr
            
        else:
            raise ValueError('Unknown method `%s`' % excitation_method)
        
        if clip:
            # limit the MI to the mixture entropy
            return np.clip(MI, 0, self.mixture_entropy())
        else:
            return MI
        
        
    def density_optimal(self, approx=True, **kwargs):
        """ return the estimated optimal activity fraction for the simple case
        where all h are the same. The estimate relies on an approximation that
        all receptors are independent and is thus independent of the number of 
        receptors. The estimate is thus only good in the limit of low Nr.
        """
        # approximate using mean substrate size
        m = self.substrate_probabilities.sum()
        
        density_opt = min(0.5 / m, 1)    

        if approx:
            return density_opt
        
        # solve a numerical equation
        obj = self.copy()
        
        def reduction(density):
            """ helper function that evaluates the mutual information """            
            obj.density = density
            return -obj.mutual_information(**kwargs)
        
        res = optimize.minimize(reduction, density_opt, bounds=[(0, 1)])
        
        if res.success:
            return res.x[0]
        else:
            raise RuntimeError(res) 
    
    
    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        return {'density': self.density_optimal(assume_homogeneous=True)}
        
        