'''
Created on Feb 22, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import numpy as np

from binary_response.sparse_mixtures.lib_spr_numeric import LibrarySparseNumeric
from .at_base import AdaptiveThresholdMixin
from utils.math.stats import StatisticsAccumulator



class AdaptiveThresholdNumeric(AdaptiveThresholdMixin, LibrarySparseNumeric):
    """ represents a single receptor library that handles sparse mixtures that
    where receptors get active if their excitation is above a fraction of the
    total excitation """
    
    
    def concentration_statistics_normalized_monte_carlo(self):
        """ determines the statistics of the normalized concentration
        numerically using a monte carlo method """
        c_hat_stats = StatisticsAccumulator()
        for ci in self._sample_mixtures():
            ctot = ci.sum()
            if ctot > 0:
                ci /= ctot
            c_hat_stats.add(ci)
            
        return {'mean': c_hat_stats.mean, 'std': c_hat_stats.std,
                'var': c_hat_stats.var}
    
    
    def _excitation_statistics_monte_carlo_base(self, ret_correlations=False):
        """ 
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        This function just calculates the statistics of unnormalized
        excitations, which is implemented in the parent function
        We implemented this as a separate function so it can selectively be
        replaced with a version that is sped up by numba         
        """
        parent = super(AdaptiveThresholdNumeric, self)
        return parent.excitation_statistics_monte_carlo(ret_correlations)
    
    
    def excitation_statistics_monte_carlo(self, ret_correlations=False,
                                          normalized=False):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        
        The algorithms used here have been taken from
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        if not normalized:
            return self._excitation_statistics_monte_carlo_base(
                                                            ret_correlations)
            
        S_ni = self.sens_mat
        S_ni_mean = S_ni.mean()

        # initialize the statistics calculation
        stats = StatisticsAccumulator(ret_cov=ret_correlations)

        # sample mixtures and safe the requested data
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            e_n /= c_i.sum() * S_ni_mean #< normalize
            stats.add(e_n)

        # return the requested statistics
        if ret_correlations:
            try:
                enm_cov = stats.cov
            except RuntimeError:
                enm_cov = np.full((self.Nr, self.Nr), np.nan, np.double)
            en_var = np.diag(enm_cov)
            return {'mean': stats.mean, 'std': np.sqrt(en_var), 'var': en_var,
                    'cov': enm_cov}
        else:        
            en_var = stats.var 
            return {'mean': stats.mean, 'std': np.sqrt(en_var), 'var': en_var}
                

    def excitation_threshold_statistics(self):
        """ returns the statistics of the excitation threshold that receptors
        have to overcome to be part of the activation pattern.
        """
        S_ni = self.sens_mat
        alpha = self.threshold_factor

        e_thresh_stats = StatisticsAccumulator()

        # iterate over samples and collect information about the threshold        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            e_thresh = alpha * e_n.mean()
            e_thresh_stats.add(e_thresh)

        return {'mean': e_thresh_stats.mean,
                'var': e_thresh_stats.var,
                'std': e_thresh_stats.std}
                    
                    
    def _sample_excitations(self, steps=None):
        """ sample excitation vectors """
        S_ni = self.sens_mat

        # iterate over mixtures and yield corresponding excitations
        for c_i in self._sample_mixtures(steps):
            yield np.dot(S_ni, c_i)
            
    
    def _sample_activities(self, steps=None):
        """ sample activity vectors """
        S_ni = self.sens_mat
        alpha = self.threshold_factor

        # iterate over mixtures and yield corresponding activities
        for c_i in self._sample_mixtures(steps):
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= alpha * e_n.mean())
            yield a_n
            
    
    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for a_n in self._sample_activities():
            r_n[a_n] += 1
            if ret_correlations:
                r_nm[np.outer(a_n, a_n)] += 1
            
        r_n /= self._sample_steps
        if ret_correlations:
            r_nm /= self._sample_steps
            return r_n, r_nm
        else:
            return r_n        


    def receptor_activity_estimate(self, ret_correlations=False,
                                   excitation_model='default', clip=False):
        """ estimates the average activity of each receptor """
        raise NotImplementedError


    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    excitation_model='default', clip=False):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        raise NotImplementedError


    def receptor_activity_for_mixture(self, c_i):
        """ returns the receptors that are activated for the mixture `c_i` """
        # calculate excitation
        e_n = np.dot(self.sens_mat, c_i)
        return (e_n >= self.threshold_factor * e_n.mean())

    
    def activation_pattern_for_mixture(self, c_i):
        """ returns the receptors that are activated for the mixture `c_i` """
        # calculate excitation
        e_n = np.dot(self.sens_mat, c_i)
        a_n = (e_n >= self.threshold_factor * e_n.mean())
        # return the indices of the active receptors
        return np.flatnonzero(a_n)
            
            
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for a_n in self._sample_activities():
            # represent activity as a single integer
            a_id = np.dot(base, a_n)
            # increment counter for this output
            count_a[a_id] += 1
            
        # count_a contains the number of times output pattern a was observed.
        # We can thus construct P_a(a) from count_a. 
        q_n = count_a / count_a.sum()
        
        # calculate the mutual information from the result pattern
        MI = -sum(q*np.log2(q) for q in q_n if q != 0)

        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        
    
    def mutual_information_estimate_fast(self):
        """ not implemented for adaptive thresholds """ 
        raise NotImplementedError
    
    
    def set_threshold_from_activity_numeric(self, activity, method='auto',
                                            steps=50, verbose=False,
                                            estimate=None):
        """ determines the threshold that leads to a given `activity`.
        
        `method` determines the method that is used to determine the receptor
            activity
        `steps` sets the number of optimization steps that are used
        `verbose` determines whether intermediate output should be printed
        `estimate` gives an estimate for the threshold_factor. A good estimate
            generally speeds up the convergence of the algorithm.
        """
        # lazy import of the Covariance Matrix Adaptation Evolution Strategy
        # package since it is only used in this method and the rest of the code
        # should be able to run without it
        import cma
        
        if not 0 < activity < 1:
            raise ValueError('Activity must be between 0 and 1')
        
        if estimate is None:
            estimate = 1
        
        def cost_function(alpha):
            """ objective function """
            self.threshold_factor = alpha.mean()
            an = self.receptor_activity(method=method).mean()
            return (an - activity)**2
        
        options = {'maxfevals': steps,
                   'bounds': [0, np.inf],
                   'verb_disp': 1 * int(verbose),
                   'verb_log': 0}
        
        # determine the correct threshold by optimization
        # we here use a two dimensional search, because this particular
        # implementation of cma is not implemented for scalar optimization.
        res = cma.fmin(cost_function, [estimate]*2, 0.1, options=options)
            
        self.threshold_factor = res[0].mean()
        return self.threshold_factor    
        