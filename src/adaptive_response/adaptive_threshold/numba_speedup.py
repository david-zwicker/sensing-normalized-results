'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import functools
import logging

import numba
import numpy as np

# these methods are used in getattr calls
from . import at_numeric
from binary_response.sparse_mixtures.numba_speedup import \
                        LibrarySparseNumeric_excitation_statistics_monte_carlo
from utils.math.distributions import lognorm_mean_var_to_mu_sigma
from utils.numba.patcher import (NumbaPatcher, check_return_value_approx,
                                 check_return_dict_approx)


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=at_numeric)



# copy the accelerated method from the binary_response package
numba_patcher.register_method(
    'AdaptiveThresholdNumeric._excitation_statistics_monte_carlo_base',
    LibrarySparseNumeric_excitation_statistics_monte_carlo,
    functools.partial(check_return_dict_approx, atol=0.1, rtol=0.1)
)


    
receptor_activity_monte_carlo_numba_template = """ 
def function(steps, alpha, S_ni, p_i, c_means, c_spread, ret_correlations, r_n,
             r_nm):
    ''' calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' '''
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)

    # sample mixtures according to the probabilities of finding ligands
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i
        
        # get excitation threshold
        e_thresh = alpha * e_n.mean()
        
        # calculate the activity pattern 
        for n in range(Nr):
            if e_n[n] >= e_thresh:
                r_n[n] += 1
                
        if ret_correlations:
            # calculate the correlations
            for n in range(Nr):
                if e_n[n] >= e_thresh:
                    r_nm[n, n] += 1
                    for m in range(n):
                        if e_n[m] >= e_thresh:
                            r_nm[n, m] += 1
                            r_nm[m, n] += 1
"""


def AdaptiveThresholdNumeric_receptor_activity_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = receptor_activity_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    # make sure all necessary objects are in the scope
    scope = {'np': np} 
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


AdaptiveThresholdNumeric_receptor_activity_monte_carlo_expon_numba = \
    AdaptiveThresholdNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
AdaptiveThresholdNumeric_receptor_activity_monte_carlo_lognorm_numba = \
    AdaptiveThresholdNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
AdaptiveThresholdNumeric_receptor_activity_monte_carlo_bernoulli_numba = \
    AdaptiveThresholdNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    
    

def AdaptiveThresholdNumeric_receptor_activity_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    fixed_mixture_size = self.parameters['fixed_mixture_size']
    if self.is_correlated_mixture or fixed_mixture_size is not None:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = AdaptiveThresholdNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)

    r_n = np.zeros(self.Nr) 
    r_nm = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        AdaptiveThresholdNumeric_receptor_activity_monte_carlo_expon_numba(
            steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
    
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        AdaptiveThresholdNumeric_receptor_activity_monte_carlo_lognorm_numba(
            steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )

    elif c_distribution == 'bernoulli':
        AdaptiveThresholdNumeric_receptor_activity_monte_carlo_bernoulli_numba(
            steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = AdaptiveThresholdNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)
    
    # return the normalized output
    r_n /= steps
    if ret_correlations:
        r_nm /= steps
        return r_n, r_nm
    else:
        return r_n


numba_patcher.register_method(
    'AdaptiveThresholdNumeric.receptor_activity_monte_carlo',
    AdaptiveThresholdNumeric_receptor_activity_monte_carlo,
    functools.partial(check_return_value_approx, rtol=0.1, atol=0.1)
)



mutual_information_monte_carlo_numba_template = ''' 
def function(steps, alpha, S_ni, p_i, c_means, c_spread, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)
    
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< reset excitation pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i => choose c_i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i

        # get excitation threshold
        e_thresh = alpha * e_n.mean()
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for n in range(Nr):
            if e_n[n] >= e_thresh:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    prob_a /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
'''


def AdaptiveThresholdNumeric_mutual_information_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = mutual_information_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    # make sure all necessary objects are in the scope
    scope = {'np': np} 
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


AdaptiveThresholdNumeric_mutual_information_monte_carlo_expon_numba = \
    AdaptiveThresholdNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
AdaptiveThresholdNumeric_mutual_information_monte_carlo_lognorm_numba = \
    AdaptiveThresholdNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
AdaptiveThresholdNumeric_mutual_information_monte_carlo_bernoulli_numba = \
    AdaptiveThresholdNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    


def AdaptiveThresholdNumeric_mutual_information_monte_carlo(
                                                self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = AdaptiveThresholdNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    prob_a = np.zeros(2**self.Nr)
    
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        MI = AdaptiveThresholdNumeric_mutual_information_monte_carlo_expon_numba(                                
            self.monte_carlo_steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )
        
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        MI = AdaptiveThresholdNumeric_mutual_information_monte_carlo_lognorm_numba(
            self.monte_carlo_steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            prob_a
        )        
        
    elif c_distribution == 'bernoulli':
        MI = AdaptiveThresholdNumeric_mutual_information_monte_carlo_bernoulli_numba(
            self.monte_carlo_steps,  
            self.threshold_factor, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )        
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = AdaptiveThresholdNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'AdaptiveThresholdNumeric.mutual_information_monte_carlo',
    AdaptiveThresholdNumeric_mutual_information_monte_carlo,
    functools.partial(check_return_value_approx, rtol=0.1, atol=0.1)
)
