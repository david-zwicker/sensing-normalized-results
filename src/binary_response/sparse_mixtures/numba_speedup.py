'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import logging
import math
import numba
import numpy as np

# these methods are used in getattr calls
from . import lib_spr_numeric
from utils.math.distributions import lognorm_mean_var_to_mu_sigma
from utils.numba.patcher import (NumbaPatcher, check_return_value_approx,
                                 check_return_value_exact,
                                 check_return_dict_approx)


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=lib_spr_numeric)



excitation_statistics_monte_carlo_numba_template = """ 
def function(steps, S_ni, p_i, c_means, c_spread, ret_correlations, en_mean,
             enm_cov):
    ''' calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' '''
    Nr, Ns = S_ni.shape
    e_n = np.empty(Nr, np.double)
    delta = np.empty(Nr, np.double)

    # sample mixtures according to the probabilities of finding ligands
    for count in range(1, steps + 1):
        # choose a mixture vector according to substrate probabilities
        e_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    e_n[n] += S_ni[n, i] * c_i
        
        # calculate the means of the excitation
        for n in range(Nr):
            delta[n] = (e_n[n] - en_mean[n]) / count
            en_mean[n] += delta[n]

        if ret_correlations:
            # calculate the full covariance matrix
            for n in range(Nr):
                for m in range(Nr):
                    enm_cov[n, m] += ((count - 1) * delta[n] * delta[m]
                                      - enm_cov[n, m] / count)
        else:
            # only calculate the variances
            for n in range(Nr):
                enm_cov[n, n] += ((count - 1) * delta[n] * delta[n]
                                  - enm_cov[n, n] / count)
                
    if steps < 2:
        enm_cov[:] = np.nan
    else:
        enm_cov *= steps / (steps - 1)
"""


def LibrarySparseNumeric_excitation_statistics_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = excitation_statistics_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    scope = {'np': np} #< make sure numpy is in the scope
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


LibrarySparseNumeric_excitation_statistics_monte_carlo_expon_numba = \
    LibrarySparseNumeric_excitation_statistics_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
LibrarySparseNumeric_excitation_statistics_monte_carlo_lognorm_numba = \
    LibrarySparseNumeric_excitation_statistics_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
LibrarySparseNumeric_excitation_statistics_monte_carlo_bernoulli_numba = \
    LibrarySparseNumeric_excitation_statistics_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    
    

def LibrarySparseNumeric_excitation_statistics_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    fixed_mixture_size = self.parameters['fixed_mixture_size']
    if self.is_correlated_mixture or fixed_mixture_size is not None:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = LibrarySparseNumeric_excitation_statistics_monte_carlo
        return this._python_function(self, ret_correlations)

    en_mean = np.zeros(self.Nr) 
    enm_cov = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        LibrarySparseNumeric_excitation_statistics_monte_carlo_expon_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            en_mean, enm_cov
        )
    
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        LibrarySparseNumeric_excitation_statistics_monte_carlo_lognorm_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            ret_correlations,
            en_mean, enm_cov
        )

    elif c_distribution == 'bernoulli':
        LibrarySparseNumeric_excitation_statistics_monte_carlo_bernoulli_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            en_mean, enm_cov
        )
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = LibrarySparseNumeric_excitation_statistics_monte_carlo
        return this._python_function(self, ret_correlations)
    
    # return the normalized output
    en_var = np.diag(enm_cov)
    if ret_correlations:
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
    else:
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var}


numba_patcher.register_method(
    'LibrarySparseNumeric.excitation_statistics_monte_carlo',
    LibrarySparseNumeric_excitation_statistics_monte_carlo,
    check_return_dict_approx
)



receptor_activity_monte_carlo_numba_template = """ 
def function(steps, S_ni, p_i, c_means, c_spread, ret_correlations, r_n, r_nm):
    ''' calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' '''
    Nr, Ns = S_ni.shape
    a_n = np.empty(Nr, np.double)

    # sample mixtures according to the probabilities of finding ligands
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        a_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    a_n[n] += S_ni[n, i] * c_i
        
        # calculate the activity pattern 
        for n in range(Nr):
            if a_n[n] >= 1:
                r_n[n] += 1
                
        if ret_correlations:
            # calculate the correlations
            for n in range(Nr):
                if a_n[n] >= 1:
                    r_nm[n, n] += 1
                    for m in range(n):
                        if a_n[m] >= 1:
                            r_nm[n, m] += 1
                            r_nm[m, n] += 1
"""


def LibrarySparseNumeric_receptor_activity_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = receptor_activity_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    scope = {'np': np} #< make sure numpy is in the scope
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


LibrarySparseNumeric_receptor_activity_monte_carlo_expon_numba = \
    LibrarySparseNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
LibrarySparseNumeric_receptor_activity_monte_carlo_lognorm_numba = \
    LibrarySparseNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
LibrarySparseNumeric_receptor_activity_monte_carlo_bernoulli_numba = \
    LibrarySparseNumeric_receptor_activity_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    
    

def LibrarySparseNumeric_receptor_activity_monte_carlo(
                                               self, ret_correlations=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    fixed_mixture_size = self.parameters['fixed_mixture_size']
    if self.is_correlated_mixture or fixed_mixture_size is not None:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = LibrarySparseNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    r_n = np.zeros(self.Nr) 
    r_nm = np.zeros((self.Nr, self.Nr)) 
    steps = self.monte_carlo_steps
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        LibrarySparseNumeric_receptor_activity_monte_carlo_expon_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
    
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        LibrarySparseNumeric_receptor_activity_monte_carlo_lognorm_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )

    elif c_distribution == 'bernoulli':
        LibrarySparseNumeric_receptor_activity_monte_carlo_bernoulli_numba(
            steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            ret_correlations,
            r_n, r_nm
        )
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = LibrarySparseNumeric_receptor_activity_monte_carlo
        return this._python_function(self, ret_correlations)
    
    # return the normalized output
    r_n /= steps
    if ret_correlations:
        r_nm /= steps
        return r_n, r_nm
    else:
        return r_n


numba_patcher.register_method(
    'LibrarySparseNumeric.receptor_activity_monte_carlo',
    LibrarySparseNumeric_receptor_activity_monte_carlo,
    check_return_value_approx
)



mutual_information_monte_carlo_numba_template = ''' 
def function(steps, S_ni, p_i, c_means, c_spread, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Nr, Ns = S_ni.shape
    a_n = np.empty(Nr, np.double)
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        a_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < p_i[i]:
                # mixture contains substrate i
                {CONCENTRATION_GENERATOR}
                for n in range(Nr):
                    a_n[n] += S_ni[n, i] * c_i
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for n in range(Nr):
            if a_n[n] >= 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize the probabilities by the number of steps we did
    for k in range(len(prob_a)):
        prob_a[k] /= steps
    
    # calculate the mutual information from the observed probabilities
    MI = 0
    for pa in prob_a:
        if pa > 0:
            MI -= pa*np.log2(pa)
    
    return MI
'''


def LibrarySparseNumeric_mutual_information_monte_carlo_numba_generator(conc_gen):
    """ generates a function that calculates the receptor activity for a given
    concentration generator """
    func_code = mutual_information_monte_carlo_numba_template.format(
        CONCENTRATION_GENERATOR=conc_gen)
    scope = {'np': np} #< make sure numpy is in the scope
    exec(func_code, scope)
    func = scope['function']
    return numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)(func)


LibrarySparseNumeric_mutual_information_monte_carlo_expon_numba = \
    LibrarySparseNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.exponential() * c_means[i]")
    
# Note that the parameter c_mean is actually the mean of the underlying normal
# distribution
LibrarySparseNumeric_mutual_information_monte_carlo_lognorm_numba = \
    LibrarySparseNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = np.random.lognormal(c_means[i], c_spread[i])")
    
LibrarySparseNumeric_mutual_information_monte_carlo_bernoulli_numba = \
    LibrarySparseNumeric_mutual_information_monte_carlo_numba_generator(
        "c_i = c_means[i]")
    


def LibrarySparseNumeric_mutual_information_monte_carlo(
                                                self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = LibrarySparseNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)

    # prevent integer overflow in collecting activity patterns
    assert self.Nr <= self.parameters['max_num_receptors'] <= 63

    prob_a = np.zeros(2**self.Nr)
 
    # call the jitted function
    c_distribution = self.parameters['c_distribution']
    if c_distribution == 'exponential':
        MI = LibrarySparseNumeric_mutual_information_monte_carlo_expon_numba(
            self.monte_carlo_steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )
        
    elif c_distribution == 'log-normal':
        mus, sigmas = lognorm_mean_var_to_mu_sigma(self.c_means, self.c_vars,
                                                   'numpy')
        MI = LibrarySparseNumeric_mutual_information_monte_carlo_lognorm_numba(
            self.monte_carlo_steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            mus, sigmas,                  #< concentration statistics
            prob_a
        )        
        
    elif c_distribution == 'bernoulli':
        MI = LibrarySparseNumeric_mutual_information_monte_carlo_bernoulli_numba(
            self.monte_carlo_steps, self.sens_mat,
            self.substrate_probabilities, #< p_i
            self.c_means, 0,              #< concentration statistics
            prob_a
        )        
        
    else:
        logging.warning('Numba code is not implemented for distribution `%s`. '
                        'Falling back to pure-python method.', c_distribution)
        this = LibrarySparseNumeric_mutual_information_monte_carlo
        return this._python_function(self, ret_prob_activity)
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI


numba_patcher.register_method(
    'LibrarySparseNumeric.mutual_information_monte_carlo',
    LibrarySparseNumeric_mutual_information_monte_carlo,
    check_return_value_approx
)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibrarySparseNumeric_mutual_information_estimate_fast_numba(
                                                     pi, c_means, c_vars, S_ni):
    """ returns a simple estimate of the mutual information for the special
    case that ret_prob_activity=False, excitation_model='default',
    mutual_information_method='default', and clip=True.
    """
    Nr, Ns = S_ni.shape
    
    en_mean = np.zeros(Nr, np.double)
    enm_cov = np.zeros((Nr, Nr), np.double)
    
    # calculate the statistics of the excitation
    for i in range(Ns):
        ci_mean = pi[i] * c_means[i]
        ci_var = pi[i] * ((1 - pi[i])*c_means[i]**2 + c_vars[i])
        
        for n in range(Nr):
            en_mean[n] += S_ni[n, i] * ci_mean
            for m in range(n + 1):
                enm_cov[n, m] += S_ni[n, i] * S_ni[m, i] * ci_var

    # calculate the receptor activity
    qn = en_mean #< reuse the memory
    for n in range(Nr):
        if en_mean[n] > 0:
            # mean is zero => qn = 0 (which we do not need to set because it is
            # already zero)
            if enm_cov[n, n] == 0:
                # variance is zero => q_n = Theta(e_n - 1)
                if en_mean[n] >= 1:
                    qn[n] = 1
                else:
                    qn[n] = 0
            else:
                # proper evaluation
                en_cv2 = enm_cov[n, n] / en_mean[n]**2
                enum = math.log(math.sqrt(1 + en_cv2) / en_mean[n])
                denom = math.sqrt(2*math.log1p(en_cv2))
                qn[n] = 0.5 * math.erfc(enum/denom)
    
    # calculate the crosstalk and the mutual information in one iteration
    prefactor = 8/math.log(2)/(2*np.pi)**2
    
    MI = 0
    for n in range(Nr):
        if 0 < qn[n] < 1:
            MI -= qn[n]*np.log2(qn[n]) + (1 - qn[n])*np.log2(1 - qn[n])
        if enm_cov[n, n] > 0: 
            for m in range(n):
                if enm_cov[m, m] > 0:
                    rho2 = enm_cov[n, m]**2 / (enm_cov[n, n] * enm_cov[m, m])
                    MI -= prefactor * rho2

    # clip the result to [0, Nr]
    if MI < 0:
        return 0
    elif MI > Nr:
        return Nr
    else:
        return MI



def LibrarySparseNumeric_mutual_information_estimate_fast(self):
    """ returns a simple estimate of the mutual information for the special
    case that ret_prob_activity=False, excitation_model='default',
    mutual_information_method='default', and clip=True.
    """
    return LibrarySparseNumeric_mutual_information_estimate_fast_numba(
        self.substrate_probabilities, self.c_means,
        self.c_vars, self.sens_mat
    )

  

numba_patcher.register_method(
    'LibrarySparseNumeric.mutual_information_estimate_fast',
    LibrarySparseNumeric_mutual_information_estimate_fast,
    check_return_value_exact
)
