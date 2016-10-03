'''
Created on Mar 27, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

General note on multiprocessing
-------------------------------
Some of the functions support multiprocessing to distribute a calculation among
several processors. If this is used, it is best practice to safe-guard the main
program with the following construct

if __name__ == '__main__': 
    <main code of the program>

This is also explained in
https://docs.python.org/2/library/multiprocessing.html#using-a-pool-of-workers
'''

from __future__ import division

import copy
import collections
import functools
import itertools
import time
import multiprocessing as mp
import random

import numpy as np
import scipy.misc
from six.moves import range, zip

from .lib_bin_base import LibraryBinaryBase
from ..library_numeric_base import LibraryNumericMixin



LN2 = np.log(2)



class LibraryBinaryNumeric(LibraryNumericMixin, LibraryBinaryBase):
    """ represents a single receptor library that handles binary mixtures """
    
    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'max_num_receptors': 28,        #< prevents memory overflows
        'max_steps': 1e7,               #< maximal number of steps 
        'sensitivity_matrix': None,     #< will be calculated if not given
        'sensitivity_matrix_params': None, #< parameters determining S_ni
        'inefficiency_weight': 1,       #< weighting parameter for inefficiency
        'brute_force_threshold_Ns': 10, #< largest Ns for using brute force 
        'monte_carlo_steps': 'auto',    #< default steps for monte carlo
        'monte_carlo_steps_min': 1e4,   #< minimal steps for monte carlo
        'monte_carlo_steps_max': 1e5,   #< maximal steps for monte carlo
        'metropolis_steps': 1e5,        #< default number of Metropolis steps
        'metropolis_steps_min': 1e4,    #< minimal steps for metropolis
        'metropolis_steps_max': 1e5,    #< maximal steps for metropolis
        'fixed_mixture_size': None,     #< fixed m or None
        'optimizer_values_count': 1024, #< maximal number of values stored 
        'anneal_Tmax': 1e-1,            #< Max (starting) temp. for annealing
        'anneal_Tmin': 1e-3,            #< Min (ending) temp. for annealing
        'verbosity': 0,                 #< verbosity level    
    }
    

    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        # the call to the inherited method also sets the default parameters from
        # this class
        super(LibraryBinaryNumeric, self).__init__(num_substrates,
                                                   num_receptors, parameters)        

        # prevent integer overflow in collecting activity patterns
        assert num_receptors <= self.parameters['max_num_receptors'] <= 63

        # check fixed_mixture_size parameter
        fixed_mixture_size = self.parameters['fixed_mixture_size']
        if fixed_mixture_size is False:
            # special case where we accept False and silently convert to None
            self.parameters['fixed_mixture_size'] = None
        elif fixed_mixture_size is not None:
            # if the value is not None it better is an integer
            try:
                fixed_mixture_size = int(fixed_mixture_size)
                if 0 <= fixed_mixture_size <= self.Ns:
                    self.parameters['fixed_mixture_size'] = fixed_mixture_size
                else:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError('`fixed_mixture_size` must either be None or '
                                 'an integer between 0 and Ns.')


    @classmethod
    def get_random_arguments(cls, fixed_mixture_size=None, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryNumeric, cls).get_random_arguments(**kwargs)
        
        if fixed_mixture_size is not None:
            args['parameters']['fixed_mixture_size'] = fixed_mixture_size

        return args


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        # create a instance with random parameters
        obj = super(LibraryBinaryNumeric, cls).create_test_instance(**kwargs)
        # choose an optimal interaction matrix
        obj.choose_sensitivity_matrix('auto')
        return obj
    
                
    def get_steps(self, scheme):
        """ calculate the number of steps to do for `scheme`"""
        if scheme == 'monte_carlo':
            # calculate the number of steps for a monte-carlo scheme
            if self.parameters['monte_carlo_steps'] == 'auto':
                steps_min = self.parameters['monte_carlo_steps_min']
                steps_max = self.parameters['monte_carlo_steps_max']
                steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
                # Here, the factor 10 is an arbitrary scaling factor
            else:
                steps = self.parameters['monte_carlo_steps']
            
        elif scheme == 'metropolis':
            # calculate the number of steps for a metropolis scheme
            if self.parameters['metropolis_steps'] == 'auto':
                steps_min = self.parameters['metropolis_steps_min']
                steps_max = self.parameters['metropolis_steps_max']
                steps = np.clip(10 * 2**self.Nr, steps_min, steps_max) 
                # Here, the factor 10 is an arbitrary scaling factor
            else:
                steps = self.parameters['metropolis_steps']
                
        else:
            raise ValueError('Unknown stepping scheme `%s`' % scheme)
            
        return int(steps)


    def choose_sensitivity_matrix(self, density=0, avoid_correlations=False):
        """ creates a interaction matrix with the given properties """
        shape = (self.Nr, self.Ns)
        
        if density == 'auto':
            # determine optimal parameters for the interaction matrix
            from binary_response.binary_mixtures.lib_bin_theory import LibraryBinaryUniform
            theory = LibraryBinaryUniform.from_other(self)
            density = theory.get_optimal_library()['density']
            
        if density == 0:
            # simple case of empty matrix
            self.sens_mat = np.zeros(shape, np.uint8)
            
        elif density >= 1:
            # simple case of full matrix
            self.sens_mat = np.ones(shape, np.uint8)
            
        elif avoid_correlations:
            # choose receptor substrate interaction randomly but try to avoid
            # correlations between the receptors
            self.sens_mat = np.zeros(shape, np.uint8)
            num_entries = int(round(density * self.Nr * self.Ns))
            
            empty_sens_mat = True
            while num_entries > 0:
                # specify the substrates that we want to detect
                if num_entries >= self.Ns:
                    i_ids = np.arange(self.Ns)
                    num_entries -= self.Ns
                else:
                    i_ids = np.random.choice(np.arange(self.Ns), num_entries,
                                             replace=False)
                    num_entries = 0
                    
                if empty_sens_mat:
                    # set the receptors for the substrates
                    a_ids = np.random.randint(0, self.Nr, len(i_ids))
                    for i, a in zip(i_ids, a_ids):
                        self.sens_mat[a, i] = 1
                    empty_sens_mat = False
                    
                else:
                    # choose receptors for each substrate from the ones that
                    # are not activated, yet
                    for i in i_ids:
                        a_ids = np.flatnonzero(self.sens_mat[:, i] == 0)
                        self.sens_mat[random.choice(a_ids), i] = 1
            
        else: # not avoid_correlations:
            # choose receptor substrate interaction randomly and don't worry
            # about correlations
            self.sens_mat = (np.random.random(shape) < density).astype(np.uint8)
            
        # save the parameters determining this matrix
        self.parameters['sensitivity_matrix_params'] = {
            'density': density,
            'avoid_correlations': avoid_correlations
        }   
            
            
    def sort_sensitivity_matrix(self, sensitivity_matrix=None):
        """ return the sorted `sensitivity_matrix` or sorts the internal
        sensitivity_matrix in place. This function rearranges receptors such
        that receptors reacting to an equal number of substrates and to similar
        substrates are close together. """
        
        if sensitivity_matrix is None:
            sens_mat = self.sens_mat
        else:
            sens_mat = sensitivity_matrix

        data = [(sum(item), list(item)) for item in sens_mat]
        sens_mat = np.array([item[1] for item in sorted(data)])

        if sensitivity_matrix is None:
            self.sens_mat = sens_mat
        else:
            return sens_mat
        
        
    @property
    def _iterate_steps(self):
        """ return the number of steps we iterate over """
        mixture_size = self.parameters['fixed_mixture_size']
        if mixture_size is None:
            return 2 ** self.Ns
        else:
            return scipy.misc.comb(self.Ns, mixture_size, exact=True)
        
        
    def _iterate_mixtures(self):
        """ iterate over all mixtures and yield the mixture with probability """
        
        if self._iterate_steps > self.parameters['max_steps']:
            raise RuntimeError('The iteration would take more than %g steps'
                               % self.parameters['max_steps'])
        
        hi = self.commonness
        Jij = self.correlations

        mixture_size = self.parameters['fixed_mixture_size']
        if mixture_size is None:
            # iterate over all mixtures
            for c in itertools.product((0, 1), repeat=self.Ns):
                c =  np.array(c, np.uint8)
                weight_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
                yield c, weight_c
                
        elif mixture_size == 0:
            # special case which is not covered by the iteration below
            yield np.zeros(self.Ns, np.uint8), 1
                
        elif mixture_size == self.Ns:
            # special case which is not covered by the iteration below
            yield np.ones(self.Ns, np.uint8), 1
                
        else:
            # iterate over all mixtures with constant number of substrates
            c = np.zeros(self.Ns, np.uint8)
            for nz in itertools.combinations(range(self.Ns), mixture_size):
                c[:] = 0
                c[np.array(nz)] = 1
                weight_c = np.exp(np.dot(np.dot(Jij, c) + hi, c))
                yield c, weight_c


    @property
    def _sample_steps(self):
        """ returns the number of steps that are sampled """
        mixture_size = self.parameters['fixed_mixture_size']
        if not self.is_correlated_mixture and mixture_size is None:
            return self.get_steps('monte_carlo')
        else:
            return self.get_steps('metropolis')


    def _sample_mixtures(self, steps=None, dtype=np.uint):
        """ sample mixtures with uniform probability yielding single mixtures """
        if steps is None:
            steps = self._sample_steps
        
        return _sample_binary_mixtures(self, steps, dtype)
                        
            
    def mixture_statistics(self, method='auto'):
        """ calculates statistics of mixtures. Returns a vector with the 
        frequencies at which substrates are present in mixtures and a matrix
        of correlations among substrates """

        if method == 'auto':
            fixed_mixture_size = self.parameters['fixed_mixture_size']
            
            if self.is_correlated_mixture or fixed_mixture_size is not None:
                # mixture has correlations => we do Metropolis sampling
                if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                    method = 'brute-force'
                else:
                    method = 'monte-carlo'
            else:
                # the estimate is exact for mixtures without correlations
                method = 'estimate'

        if method == 'brute-force'  or method == 'brute_force':
            return self.mixture_statistics_brute_force()
        elif method == 'monte-carlo'  or method == 'monte_carlo':
            return self.mixture_statistics_monte_carlo()
        elif method == 'estimate':
            return self.mixture_statistics_estimate()
        else:
            raise ValueError('Unknown method `%s` for mixture statistics'
                             % method)
    
    
    def mixture_statistics_brute_force(self):
        """ calculates mixture statistics using a brute force algorithm """
        
        Z = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))
        
        # iterate over all mixtures
        for c, weight_c in self._iterate_mixtures():
            Z += weight_c        
            hist1d += c * weight_c
            hist2d += np.outer(c, c) * weight_c
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d / Z
        cij = hist2d / Z
        cij_corr = cij - np.outer(ci_mean, ci_mean)
        
        ci_var = np.diag(cij_corr)
        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': cij_corr}
    
    
    def mixture_statistics_monte_carlo(self):
        """ calculates mixture statistics using a metropolis algorithm """
        return self.concentration_statistics_monte_carlo()
    
    
    def mixture_statistics_estimate(self):
        """ estimates the mixture statistics """
        ci_mean = self.substrate_probabilities
        
        if self.is_correlated_mixture:
            J_ij = self.correlations
            pi_s = ci_mean
            bar_pi_s = 1 - pi_s
            
            ci_mean = pi_s * (1 + 2*bar_pi_s*np.dot(J_ij, pi_s))
            ci_var = ci_mean * (1 - ci_mean)
            cij_cov = (
                np.diag(ci_var)
                + 2*np.einsum('ij,i,j->ij', J_ij, ci_var, ci_var)
            )

        else:
            # uncorrelated mixtures
            ci_var = ci_mean * (1 - ci_mean)
            cij_cov = np.diag(ci_var)

        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': cij_cov}
         
    
    def mixture_entropy(self):
        """ return the entropy in the mixture distribution """
        
        mixture_size = self.parameters['fixed_mixture_size']
                
        if self.is_correlated_mixture or mixture_size is not None:
            # complicated case => run brute force or monte carlo
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                return self.mixture_entropy_brute_force()
            else:
                return self.mixture_entropy_monte_carlo()
        
        else:
            # simple case => calculate explicitly
            return super(LibraryBinaryNumeric, self).mixture_entropy()
        
    
    def mixture_entropy_brute_force(self):
        """ gets the entropy in the mixture distribution using brute force """
        Z, sum_wlogw = 0, 0

        # Naive implementation of measuring the entropy is
        #    p(c) = w(c) / Z   with Z = sum_c w(c)
        #    H_c  = -sum_c p(c) * log2(p(c))
        # This can be transformed to a more stable implementation:
        #    H_c = log2(Z) - 1/Z * sum_c w(c) * log2(w(c))
        
        for _, weight_c in self._iterate_mixtures():
            if weight_c > 0:
                Z += weight_c
                sum_wlogw += weight_c * np.log2(weight_c)
                
        if Z == 0:
            return 0
        else:
            return np.log2(Z) - sum_wlogw / Z
            

    def mixture_entropy_monte_carlo(self):
        """ gets the entropy in the mixture distribution using brute force """
        if self.Ns > 63:
            raise ValueError('Mixture entropy estimation only works for fewer '
                             'than 64 substrates.')
        
        # sample mixtures
        base = 2 ** np.arange(0, self.Ns)
        observations = collections.Counter()
        for c in self._sample_mixtures():
            observations[np.dot(c, base)] += 1
        
        # estimate entropy from the histogram
        counts = np.fromiter(observations.values(), np.double,
                             len(observations))
        
        # Naive implementation of measuring the entropy is
        #    ps = counts / self._sample_steps
        #    H = -np.sum(ps * np.log2(ps))
        # This can be transformed to a more stable implementation:
        log_steps = np.log2(self._sample_steps)
        return -np.sum(counts*(np.log2(counts) - log_steps))/self._sample_steps
    
    
    def receptor_crosstalk(self, method='auto', ret_receptor_activity=False,
                           **kwargs):
        """ calculates the average activity of the receptor as a response to 
        single ligands.
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'estimate':
            # estimate receptor crosstalk directly
            q_nm = self.receptor_crosstalk_estimate(**kwargs)
            if ret_receptor_activity:
                q_n = self.receptor_activity_estimate(**kwargs)
        
        else:
            # calculate receptor crosstalk from the observed probabilities
            r_n, r_nm = self.receptor_activity(method, ret_correlations=True,
                                               **kwargs)
            q_n = r_n
            q_nm = r_nm - np.outer(r_n, r_n)
            if kwargs.get('clip', False):
                np.clip(q_nm, 0, 1, q_nm)
            
        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm
    
    
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    approx_prob=False, clip=False,
                                    ignore_correlations=False):
        """ estimates the average activity of the receptor as a response to 
        single ligands. 
        `ret_receptor_activity` determines whether the mean receptor activity
            will also be returned.
        `approx_prob` determines whether the probabilities of encountering
            ligands in mixtures are calculated exactly or only approximative,
            which should work for small probabilities.
        `clip` determines whether the estimates will be forced to be in [0, 1].
        `ignore_correlations` determines whether correlations in the mixtures
            will be ignored or not.
        """
        if not ignore_correlations and self.is_correlated_mixture:
            r_n, r_nm = self.receptor_activity_estimate(ret_correlations=True,
                                                        approx_prob=approx_prob,
                                                        clip=clip)
            q_nm = r_nm - np.outer(r_n, r_n)
            if clip:
                np.clip(q_nm, 0, 1, q_nm)
            if ret_receptor_activity:
                return r_n, q_nm
            else:
                return q_nm
            raise NotImplementedError('Not implemented for correlated mixtures')

        S_ni = self.sens_mat
        p_i = self.substrate_probabilities
        
        if approx_prob:
            # approximate calculation for small p_i
            q_nm = np.einsum('ni,mi,i->nm', S_ni, S_ni, p_i)
            if clip:
                np.clip(q_nm, 0, 1, q_nm)
            
        else:
            # proper calculation of the probabilities
            S_ni_mask = S_ni.astype(np.bool)
            q_nm = np.zeros((self.Nr, self.Nr))
            for n in range(self.Nr):
                for m in range(self.Nr):
                    mask = S_ni_mask[n, :] * S_ni_mask[m, :]
                    q_nm[n, m] = 1 - np.product(1 - p_i[mask])
                    
                    
        if ret_receptor_activity:
            q_n = self.receptor_activity_estimate(approx_prob=approx_prob,
                                                  clip=clip)
            return q_n, q_nm
        else:
            return q_nm
                

    def receptor_activity(self, method='auto', ret_correlations=False, **kwargs):
        """ calculates the average activity of each receptor
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        """
        if method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                method = 'brute_force'
            else:
                method = 'monte_carlo'
                
        if method == 'brute_force' or method == 'brute-force':
            return self.receptor_activity_brute_force(ret_correlations, **kwargs)
        elif method == 'monte_carlo' or method == 'monte-carlo':
            return self.receptor_activity_monte_carlo(ret_correlations, **kwargs)
        elif method == 'estimate':
            return self.receptor_activity_estimate(ret_correlations, **kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)

        
    def receptor_activity_brute_force(self, ret_correlations=False):
        """ calculates the average activity of each receptor """
        S_ni = self.sens_mat
        Z = 0
        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        # iterate over all mixtures
        for c, prob_c in self._iterate_mixtures():
            # get the activity vector associated with m
            a_n = (np.dot(S_ni, c) >= 1)
            Z += prob_c

            r_n[a_n] += prob_c
            if ret_correlations:
                r_nm[np.outer(a_n, a_n)] += prob_c
                
        # return the normalized output
        r_n /= Z
        if ret_correlations:
            r_nm /= Z
            return r_n, r_nm
        else:
            return r_n
 
 
    def receptor_activity_estimate(self, ret_correlations=False,
                                   approx_prob=False, clip=False):
        """ estimates the average activity of each receptor.
        `ret_correlations` determines whether the correlations between receptors
            are returned in addition to the mean activations.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities.
        `clip` determines whether the estimates will be forced to be in [0, 1].
        """
        S_ni = self.sens_mat
        p_i = self.substrate_probabilities

        # calculate receptor activity assuming uncorrelated mixtures         
        if approx_prob:
            # approximate calculation for small p_i
            r_n = np.dot(S_ni, p_i)
            if clip:
                np.clip(r_n, 0, 1, r_n)
             
        else:
            # proper calculation of the probabilities
            r_n = np.zeros(self.Nr)
            S_ni_mask = S_ni.astype(np.bool)
            for n in range(self.Nr):
                r_n[n] = 1 - np.product(1 - p_i[S_ni_mask[n, :]])
         
        if self.is_correlated_mixture:
            # add linear correction term for correlated mixtures
            J_ij = self.correlations
            p_ni = p_i[None, :] * (1 - S_ni)
            
            corr1 = 1 + np.einsum('ij,ni,nj->n', J_ij, p_ni, p_ni)
            corr2 = 1 + np.einsum('ij,i,j->', J_ij, p_i, p_i)
            
            barr_n_0 = 1 - r_n
            barr_n = barr_n_0 * (1 + corr1 - corr2)
            r_n = 1 - barr_n
            if clip:
                np.clip(r_n, 0, 1, r_n)
         
        if ret_correlations:
            # estimate the correlations from the estimated crosstalk
            q_nm = self.receptor_crosstalk_estimate(approx_prob=approx_prob,
                                                    ignore_correlations=True)
             
            if approx_prob:
                r_nm = np.outer(r_n, r_n) + q_nm
            else:
                r_nm = 1 - (1 - q_nm)*(1 - np.outer(r_n, r_n))
                 
            if self.is_correlated_mixture:
                # add one correction term for correlated mixtures
                p_nmi = np.einsum('i,ni,mi->nmi', p_i, 1 - S_ni, 1 - S_ni)
                
                corr1 = 1 + np.einsum('ij,nmi,nmj->nm', J_ij, p_nmi, p_nmi)
                # corr2 = 1 + np.einsum('ij,i,j->', J_ij, p_i, p_i)
                # this term has already been calculated above and can be reused

                # convert r_nm_0 (here given as r_nm) into barr_nm_0
                barr_nm_0 = 1 - barr_n_0[:, None] - barr_n_0[None, :] + r_nm
                
                # correct barr_nm for the correlations J_ij
                barr_nm = barr_nm_0 * (1 + corr1 - corr2)

                # convert barr_nm into r_nm
                r_nm = 1 - barr_n[:, None] - barr_n[None, :] + barr_nm
         
            if clip:
                np.clip(r_nm, 0, 1, r_nm)
 
            return r_n, r_nm
         
        else:
            return r_n


    def mutual_information(self, excitation_method='auto', **kwargs):
        """ calculate the mutual information.
        
        `excitation_method` can be ['brute_force', 'monte_carlo', 'estimate', 
            'auto'] If it is 'auto' than the excitation_method is chosen
            automatically based on the problem size.
        `ret_prob_activity` determines whether the probabilities of the
            different outputs are returned or not
        """
        if excitation_method == 'auto':
            if self.Ns <= self.parameters['brute_force_threshold_Ns']:
                excitation_method = 'brute_force'
            else:
                excitation_method = 'monte_carlo'
                
        if excitation_method == 'brute_force' or excitation_method == 'brute-force':
            return self.mutual_information_brute_force(**kwargs)
        elif excitation_method == 'monte_carlo' or excitation_method == 'monte-carlo':
            return self.mutual_information_monte_carlo(**kwargs)
        elif excitation_method == 'estimate':
            return self.mutual_information_estimate(**kwargs)
        else:
            raise ValueError('Unknown excitation_method `%s`.' % excitation_method)
            
            
    def mutual_information_brute_force(self, ret_prob_activity=False):
        """ calculate the mutual information by constructing all possible
        mixtures """
        base = 2 ** np.arange(0, self.Nr)

        # prob_a contains the probability of finding activity a as an output.
        prob_a = np.zeros(2**self.Nr)
        for c, prob_c in self._iterate_mixtures():
            # get the associated output ...
            a = np.dot(self.sens_mat, c).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)

            prob_a[a] += prob_c
            
        # normalize the output to make it a probability distribution
        prob_a /= prob_a.sum()
        
        # calculate the mutual information
        MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
        
        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI
            
        
    def mutual_information_monte_carlo_extrapolate(self, ret_prob_activity=False):
        """ calculate the mutual information using a Monte Carlo strategy. """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
                
        base = 2 ** np.arange(0, self.Nr)
        prob_s = self.substrate_probabilities

        max_steps = self._sample_steps
        steps, MIs = [], []

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        step_check = 10000
        for step in range(max_steps):
            # choose a mixture vector according to substrate probabilities
            m = (np.random.random(self.Ns) < prob_s)
            
            # get the associated output ...
            a = np.dot(self.sens_mat, m).astype(np.bool)
            # ... and represent it as a single integer
            a = np.dot(base, a)
            # increment counter for this output
            count_a[a] += 1

            if step == step_check - 1:
                # do an extrapolation step
                # calculate the mutual information from the result pattern
                prob_a = count_a / step
                MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)
                
                # save the data                
                steps.append(step)
                MIs.append(MI)
                
                # do the extrapolation
                if len(steps) >= 3:
                    a2, a1, a0 = MIs[-3:]
                    MI_ext = (a0*a2 - a1*a1)/(a0 - 2*a1 + a2)
#                     MI_ext = self._get_extrapolated_mutual_information(steps, MIs)
                    print((step, MIs[-1], MI_ext))
                    
                step_check += 10000
            
        else:
            # count_a contains the number of times output pattern a was observed.
            # We can thus construct P_a(a) from count_a. 
            
            # calculate the mutual information from the result pattern
            prob_a = count_a / step
            MI = -sum(pa*np.log2(pa) for pa in prob_a if pa != 0)

        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI

                    
    def mutual_information_estimate(self, approx_prob=False):
        """ returns a simple estimate of the mutual information.
        `approx_prob` determines whether the probabilities of encountering
            substrates in mixtures are calculated exactly or only approximative,
            which should work for small probabilities. """
 
        # this might be not the right approach
        q_n = self.receptor_activity_estimate(approx_prob=approx_prob)
        q_nm = self.receptor_crosstalk_estimate(approx_prob=approx_prob)
                     
        # calculate the approximate mutual information
        return self._estimate_MI_from_q_values(q_n, q_nm)
        
        
    def receptor_score(self, method='auto', multiprocessing=False):
        """ calculates the usefulness of each receptor, measured by how much
        information it adds to the total mutual information.
            `method` determines which method is used to determine the mutual
                information.
            `multiprocessing` determines whether multiprocessing is used for
                determining the mutual informations of all subsystems.
        """
        init_arguments = self.init_arguments
        init_arguments['parameters']['initialize_state']['sensitivity'] = 'exact'
        init_arguments['parameters']['sensitivity_matrix'] = self.sens_mat
        joblist = [(copy.deepcopy(self.init_arguments), 'mutual_information',
                    {'method': method})]
    
        # add one job for each receptor
        for n in range(self.Nr):
            init_arguments = self.init_arguments
            init_arguments['num_receptors'] -= 1
            
            # modify the current state and add it to the job list
            sens_mat = np.delete(self.sens_mat, n, axis=0)
            init_arguments['parameters']['sensitivity_matrix'] = sens_mat
            joblist.append((copy.deepcopy(init_arguments), 'mutual_information',
                            {'method': method}))
                
        if multiprocessing:
            # calculate all results in parallel
            pool = mp.Pool(processes=self.get_number_of_cores())
            results = pool.map(_run_job, joblist)
        
        else:
            # create a generator over which we iterate later
            results = [_run_job(job) for job in joblist]
        
        # find the scores of all receptors
        scores = results[0] - np.array(results[1:])
        return scores

        
    def optimize_library(self, target, method='descent', direction='max',
                         **kwargs):
        """ optimizes the current library to maximize the result of the target
        function. By default, the function returns the best value and the
        associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try 
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function

        `method` determines the method used for optimization. Supported are
            `descent`: simple gradient descent for `steps` number of steps
            `descent_multiple`: gradient descent starting from multiple initial
                conditions.
            `anneal`: simulated annealing
        """
        if method == 'descent':
            return self.optimize_library_descent(target, direction, **kwargs)
        elif method == 'descent_multiple' or method == 'descent-multiple':
            return self.optimize_library_descent_multiple(target, direction,
                                                          **kwargs)
        elif method == 'anneal':
            return self.optimize_library_anneal(target, direction, **kwargs)
            
        else:
            raise ValueError('Unknown optimization method `%s`' % method)
            
        
    def optimize_library_descent(self, target, direction='max', steps=100,
                                 multiprocessing=False, ret_info=False,
                                 args=None):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent. By default, the function returns the
        best value and the associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try
        `multiprocessing` is a flag deciding whether multiple processes are used
            to calculate the result. Note that this has an overhead and might
            actually decrease overall performance for small problems
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function
        """
        # get the target function to call
        target_function = getattr(self, target)
        if args is not None:
            target_function = functools.partial(target_function, **args)

        # initialize the optimizer
        value = target_function()
        value_best, state_best = value, self.sens_mat.copy()
        
        if ret_info:
            # store extra information
            start_time = time.time()
            info = {'values': {}}
            values_count = self.parameters['optimizer_values_count']
            values_step = max(1, steps // values_count)
        
        if multiprocessing:
            # run the calculations in multiple processes
            pool_size = self.get_number_of_cores()
            pool = mp.Pool(processes=pool_size)
            if ret_info:
                values_step = max(1, values_step // pool_size)
            
            # iterate for given number of steps
            for step in range(int(steps) // pool_size):
                joblist = []
                init_arguments = self.init_arguments
                for _ in range(pool_size):
                    # modify the current state and add it to the job list
                    i = random.randrange(self.sens_mat.size)
                    self.sens_mat.flat[i] = 1 - self.sens_mat.flat[i]
                    params = init_arguments['parameters'] 
                    params['sensitivity_matrix'] = self.sens_mat
                    params['initialize_state']['sensitivity'] = 'exact'
                    
                    joblist.append((copy.deepcopy(init_arguments), target))
                    self.sens_mat.flat[i] = 1 - self.sens_mat.flat[i]
                    
                # run all the jobs
                results = pool.map(_run_job, joblist)
                
                # find the best result  
                if direction == 'max':
                    res_best = np.argmax(results)
                    if results[res_best] > value_best:
                        value_best = results[res_best]
                        state_best = joblist[res_best][0]['parameters']['sensitivity_matrix']
                        # use the best state as a basis for the next iteration
                        self.sens_mat = state_best
                        
                elif direction == 'min':
                    res_best = np.argmin(results)
                    if results[res_best] < value_best:
                        value_best = results[res_best]
                        state_best = joblist[res_best][0]['parameters']['sensitivity_matrix']
                        # use the best state as a basis for the next iteration
                        self.sens_mat = state_best
                        
                else:
                    raise ValueError('Unsupported direction `%s`' % direction)
                        
                if ret_info and step % values_step == 0:
                    info['values'][step * pool_size] = results[res_best]
                
        else:
            # run the calculations in this process
            for step in range(int(steps)):
                # modify the current state
                i = random.randrange(self.sens_mat.size)
                self.sens_mat.flat[i] = 1 - self.sens_mat.flat[i]
    
                # get the value of the new state
                value = target_function()
                
                improved = ((direction == 'max' and value > value_best) or
                            (direction == 'min' and value < value_best))
                if improved:
                    # save the state as the new best value
                    value_best, state_best = value, self.sens_mat.copy()
                else:
                    # undo last change
                    self.sens_mat.flat[i] = 1 - self.sens_mat.flat[i]
                    
                if ret_info and step % values_step == 0:
                    info['values'][step] = value_best

        # sort the best state and store it in the current object
        state_best = self.sort_sensitivity_matrix(state_best)
        self.sens_mat = state_best.copy()

        if ret_info:
            info['total_time'] = time.time() - start_time    
            info['states_considered'] = steps
            info['performance'] = steps / info['total_time']
            return value_best, state_best, info
        else:
            return value_best, state_best
 
             
    def optimize_library_descent_multiple(self, target, direction='max',
                                          trials=4, multiprocessing=False,
                                          ret_error=False, **kwargs):
        """ optimizes the current library to maximize the result of the target
        function using gradient descent from `trials` different staring
        positions. Only the result from the best run will be returned """
        
        # pass some parameters down to the optimization function to call
        kwargs['target'] = target
        kwargs['direction'] = direction
        
        # initialize the list of jobs with an optimization job starting from the
        # current interaction matrix
        joblist = [(self.init_arguments, 'optimize_library_descent', kwargs)]
        sens_mat = self.sens_mat #< store matrix to restore it later

        # set the ensemble of sensitivity matrices to try
        self.choose_sensitivity_matrix(density='auto')
        self.parameters['initialize_state']['sensitivity'] = 'ensemble'

        # add additional jobs with random initial interaction matrices
        init_arguments = self.init_arguments
        for _ in range(trials - 1):
            joblist.append((copy.deepcopy(init_arguments),
                            'optimize_library_descent', kwargs))
            
        # restore interaction matrix of this object
        self.sens_mat = sens_mat
        
        if multiprocessing:
            # calculate all results in parallel
            pool = mp.Pool(processes=self.get_number_of_cores())
            result_iter = pool.imap_unordered(_run_job, joblist)
        
        else:
            # create a generator over which we iterate later
            result_iter = (_run_job(job) for job in joblist)
        
        # find the best result by iterating over all results
        result_best, values = None, []
        for result in result_iter:
            values.append(result[0])
            # check whether this run improved the result
            if result_best is None:
                result_best = result
            elif ((direction == 'max' and result[0] > result_best[0]) or
                  (direction == 'min' and result[0] < result_best[0])):
                result_best = result
                
        # sort the best state and store it in the current object
        state = self.sort_sensitivity_matrix(result_best[1])
        self.sens_mat = state.copy()

        if ret_error:
            # replace the best value by a tuple of the best value and its error
            value_best = result_best[0]
            value_err = np.abs(value_best - np.median(values))
            result_best = ((value_best, value_err), ) + result_best[1:]
        return result_best
                               
    
    def optimize_library_anneal(self, target, direction='max', steps=100,
                                ret_info=False, args=None):
        """ optimizes the current library to maximize the result of the target
        function using simulated annealing. By default, the function returns the
        best value and the associated interaction matrix as result.        
        
        `direction` is either 'min' or 'max' and determines whether a minimum
            or a maximum is sought.
        `steps` determines how many optimization steps we try
        `ret_info` determines whether extra information is returned from the
            optimization 
        `args` is a dictionary of additional arguments that is passed to the
            target function
        """
        # lazy import
        from .optimizer import ReceptorOptimizerAnnealer  # @UnresolvedImport
        
        # prepare the class that manages the simulated annealing
        annealer = ReceptorOptimizerAnnealer(self, target, direction, args,
                                             ret_info=ret_info)
        annealer.steps = int(steps)
        annealer.Tmax = self.parameters['anneal_Tmax']
        annealer.Tmin = self.parameters['anneal_Tmin']
        if self.parameters['verbosity'] == 0:
            annealer.updates = 0

        # do the optimization
        MI, state = annealer.optimize()

        # sort the best state and store it in the current object
        state = self.sort_sensitivity_matrix(state)
        self.sens_mat = state.copy()
        
        if ret_info:
            return MI, state, annealer.info
        else:
            return MI, state    



def _sample_binary_mixtures(model, steps, dtype=np.uint):
    """ generator function that samples mixtures according to the `model`.
        `steps` determines how many mixtures are sampled
        `dtype` determines the dtype of the resulting concentration vector
    """
    mixture_size = model.parameters['fixed_mixture_size']
            
    if not model.is_correlated_mixture and mixture_size is None:
        # use simple monte carlo algorithm
        prob_s = model.substrate_probabilities
        
        for _ in range(int(steps)):
            # choose a mixture vector according to substrate probabilities
            yield (np.random.random(model.Ns) < prob_s).astype(dtype)

    elif mixture_size is None:
        # go through all mixtures and don't keep the size constant

        # use metropolis algorithm
        hi = model.commonness
        Jij = model.correlations
        
        # start with a random concentration vector 
        c = np.random.random_integers(0, 1, model.Ns).astype(dtype)
        E_last = -np.dot(np.dot(Jij, c) + hi, c)
        
        for _ in range(int(steps)):
            i = random.randrange(model.Ns)
            c[i] = 1 - c[i] #< switch the entry
            Ei = -np.dot(np.dot(Jij, c) + hi, c)
            if Ei < E_last or random.random() < np.exp(E_last - Ei):
                # accept the new state
                E_last = Ei
            else:
                # reject the new state and revert to the last one
                c[i] = 1 - c[i]
        
            yield c
            
    elif mixture_size == 0:
        # special case which is not covered by the iteration below
        c_zero = np.zeros(model.Ns, dtype)
        for _ in range(model._sample_steps):
            yield c_zero

    elif mixture_size == model.Ns:
        # special case which is not covered by the iteration below
        c_ones = np.ones(model.Ns, dtype)
        for _ in range(steps):
            yield c_ones
                    
    else:
        # go through mixtures with keeping their size constant

        # use metropolis algorithm
        hi = model.commonness
        Jij = model.correlations

        # create random concentration vector with fixed substrate count
        c = np.r_[np.ones(mixture_size, dtype),
                  np.zeros(model.Ns - mixture_size, dtype)]
        np.random.shuffle(c)
        E_last = -np.dot(np.dot(Jij, c) + hi, c)
        
        for _ in range(int(steps)):
            # find the next mixture by swapping two items
            i0 = random.choice(np.flatnonzero(c == 0)) #< find 0
            i1 = random.choice(np.flatnonzero(c))      #< find 1
            c[i0], c[i1] = 1, 0 #< swap entries
            Ei = -np.dot(np.dot(Jij, c) + hi, c)
            if Ei < E_last or random.random() < np.exp(E_last - Ei):
                # accept the new state
                E_last = Ei
            else:
                # reject the new state and revert to the last one
                c[i0], c[i1] = 0, 1
        
            yield c



def _run_job(args):
    """ helper function for optimizing the receptor library using
    multiprocessing """
    # Note that we do not set the seed of the random number generator because
    # we already modified the interaction matrix before calling this function
    # and it does not harm us when all sub processes have the same sequence of
    # random numbers.
    
    # create the object ...
    obj = LibraryBinaryNumeric(**args[0])
    # ... get the method to evaluate ...
    method = getattr(obj, args[1])
    # ... and evaluate it
    if len(args) > 2:
        return method(**args[2])
    else:
        return method()

   
   
def performance_test(Ns=15, Nr=3):
    """ test the performance of the brute force and the Monte Carlo method """
    num = 2**Ns
    hs = np.random.random(Ns)
    model = LibraryBinaryNumeric(Ns, Nr, hs)
    
    start = time.time()
    model.mutual_information_brute_force()
    time_brute_force = time.time() - start
    print('Brute force: %g sec' % time_brute_force)
    
    start = time.time()
    model.mutual_information_monte_carlo(num)
    time_monte_carlo = time.time() - start
    print('Monte carlo: %g sec' % time_monte_carlo)
    
        
            
if __name__ == '__main__':
    performance_test()
    
