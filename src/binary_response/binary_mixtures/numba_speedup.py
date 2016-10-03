'''
Created on Jan 16, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>

Module that monkey patches classes in other modules with equivalent, but faster
methods.
'''

from __future__ import division

import logging

import numba
import numpy as np

# these methods are used in getattr calls
from . import lib_bin_numeric
from utils.numba.patcher import (NumbaPatcher, check_return_value_approx,
                                 check_return_value_exact)


NUMBA_NOPYTHON = True #< globally decide whether we use the nopython mode
NUMBA_NOGIL = True

# initialize the numba patcher and add methods one by one
numba_patcher = NumbaPatcher(module=lib_bin_numeric)



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _mixture_energy(ci, hi, Jij):
    """ helper function that calculates the "energy" associated with the
    mixture `ci`, given commonness vector `hi` and correlation matrix `Jij` """ 
    energy = 0
    Ns = ci.size
    for i in range(Ns):
        if ci[i] > 0:
            energy -= hi[i] + Jij[i, i]
            for j in range(i + 1, Ns):
                energy -= 2 * Jij[i, j] * ci[j]
    return energy


    
@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _mixture_energy_indices(indices, hi, Jij):
    """ helper function that calculates the "energy" associated with the
    mixture defined by the `indices` of substrates that are present, given
    commonness vector `hi` and correlation matrix `Jij` """ 
    energy = 0
    for k, i in enumerate(indices):
        energy -= hi[i] + Jij[i, i]
        for j in indices[k + 1:]:
            energy -= 2 * Jij[i, j]
    return energy



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _activity_pattern(ci, sens_mat):
    """ helper function that calculates the id of the activity pattern from a
    given concentration vector and the associated interaction matrix """
    Nr, Ns = sens_mat.shape
    
    # calculate the activity pattern id for given mixture `ci`
    a_id, base = 0, 1
    for n in range(Nr):
        for i in range(Ns):
            if ci[i] * sens_mat[n, i] == 1:
                # substrate is present and excites receptor
                a_id += base
                break
        base *= 2
    return a_id



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _activity_pattern_indices(indices, sens_mat):
    """ helper function that calculates the id of the activity pattern from a
    concentration vector given by the indices of ligands that are present and
    the associated interaction matrix """
    Nr = len(sens_mat)
    
    # calculate the activity pattern id for given mixture `ci`
    a_id, base = 0, 1
    for n in range(Nr):
        for i in indices:
            if sens_mat[n, i] == 1:
                # substrate is present and excites receptor
                a_id += base
                break
        base *= 2
    return a_id



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _get_entropy(data):
    """ helper function that calculates the entropy of the input array """
    H = 0
    for value in data:
        if value > 0: 
            H -= value * np.log2(value)
    return H


    
@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _get_entropy_normalize(data):
    """ helper function that calculates the entropy of the input array after
    normalizing it """
    # normalize the probabilities and calculate the entropy simultaneously
    Z = data.sum()
    H = 0
    for k in range(len(data)):
        # do in-place division such that data is normalized after this function 
        data[k] /= Z
        value = data[k]
        if value > 0: 
            H -= value * np.log2(value)
    return H

    
#===============================================================================
# ACTIVITY SINGLE
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_receptor_activity_brute_force_corr_numba(
        S_ni, hi, Jij, ret_correlations, q_n, q_nm):
    """ calculates the average activity of each receptor """
    Nr, Ns = S_ni.shape
    c_i = np.empty(Ns, np.uint)
    a_n = np.empty(Nr, np.uint)
    
    # iterate over all mixtures c
    Z = 0
    for c in range(2**Ns):
        # extract the mixture and the activity from the single integer `c`
        a_n[:] = 0
        for i in range(Ns):
            c_i[i] = c % 2
            c //= 2
        
            if c_i[i] == 1:
                # determine which receptors this substrate activates
                for n in range(Nr):
                    if S_ni[n, i] == 1:
                        a_n[n] = 1
        
        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy(c_i, hi, Jij))
        Z += pm
        
        # add probability to the active receptors
        for n in range(Nr):
            if a_n[n] >= 1:
                q_n[n] += pm
                
        if ret_correlations:
            for n in range(Nr):
                if a_n[n] >= 1:
                    q_nm[n, n] += pm
                    for m in range(n + 1, Nr):
                        if a_n[m] >= 1:
                            q_nm[n, m] += pm
                            q_nm[m, n] += pm
                
        
    # normalize by partition sum
    for n in range(Nr):        
        q_n[n] /= Z
    if ret_correlations:
        for n in range(Nr):        
            for m in range(Nr):        
                q_nm[n, m] /= Z



@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_receptor_activity_brute_force_numba(
        S_ni, p_i, ret_correlations, q_n, q_nm):
    """ calculates the average activity of each receptor """
    Nr, Ns = S_ni.shape
    a_n = np.empty(Nr, np.uint)
    
    # iterate over all mixtures m
    for m in range(2**Ns):
        pm = 1     #< probability of finding this mixture
        a_n[:] = 0  #< activity pattern of this mixture
        
        # iterate through substrates in the mixture
        for i in range(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= p_i[i]
                for n in range(Nr):
                    if S_ni[n, i] == 1:
                        a_n[n] = 1
            else:
                # substrate i is not present
                pm *= 1 - p_i[i]
                
        # add probability to the active receptors
        for n in range(Nr):
            if a_n[n] >= 1:
                q_n[n] += pm
                
        if ret_correlations:
            for n in range(Nr):
                if a_n[n] >= 1:
                    q_nm[n, n] += pm
                    for m in range(n + 1, Nr):
                        if a_n[m] >= 1:
                            q_nm[n, m] += pm
                            q_nm[m, n] += pm



def LibraryBinaryNumeric_receptor_activity_brute_force(self,
                                                       ret_correlations=False):
    """ calculates the average activity of each receptor """
    if self.parameters['fixed_mixture_size'] is not None:
        raise NotImplementedError

    q_n = np.zeros(self.Nr)
    q_nm = np.zeros((self.Nr, self.Nr))    
    
    if self.is_correlated_mixture:
        # call the jitted function for correlated mixtures
        LibraryBinaryNumeric_receptor_activity_brute_force_corr_numba(
            self.sens_mat,
            self.commonness, self.correlations, #< hi, Jij
            ret_correlations,
            q_n, q_nm
        )
        
    else:
        # call the jitted function for uncorrelated mixtures
        LibraryBinaryNumeric_receptor_activity_brute_force_numba(
            self.sens_mat,
            self.substrate_probabilities, #< p_i
            ret_correlations,
            q_n, q_nm
        )
        
    if ret_correlations:
        return q_n, q_nm
    else:
        return q_n



numba_patcher.register_method(
    'LibraryBinaryNumeric.receptor_activity_brute_force',
    LibraryBinaryNumeric_receptor_activity_brute_force,
)
    
    
#===============================================================================
# ACTIVITY CORRELATIONS
#===============================================================================


# @numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
# def LibraryBinaryNumeric_activity_correlations_brute_force_numba(
#         Ns, Nr, sens_mat, prob_s, ak, prob_a):
#     """ calculates the correlations between receptor activities """
#     # iterate over all mixtures m
#     for m in range(2**Ns):
#         pm = 1     #< probability of finding this mixture
#         ak[:] = 0  #< activity pattern of this mixture
#         
#         # iterate through substrates in the mixture
#         for i in range(Ns):
#             r = m % 2
#             m //= 2
#             if r == 1:
#                 # substrate i is present
#                 pm *= prob_s[i]
#                 for a in range(Nr):
#                     if sens_mat[a, i] == 1:
#                         ak[a] = 1
#             else:
#                 # substrate i is not present
#                 pm *= 1 - prob_s[i]
#                 
#         # add probability to the active receptors
#         for a in range(Nr):
#             if ak[a] == 1:
#                 prob_a[a, a] += pm
#                 for b in range(a + 1, Nr):
#                     if ak[b] == 1:
#                         prob_a[a, b] += pm
#                         prob_a[b, a] += pm
#                     
#     
#     
# def LibraryBinaryNumeric_activity_correlations_brute_force(self):
#     """ calculates the correlations between receptor activities """
#     if self.is_correlated_mixture:
#         raise NotImplementedError('Not implemented for correlated mixtures')
# 
#     if self.parameters['fixed_mixture_size'] is not None:
#         raise NotImplementedError('Not implemented for fixed mixtures')
#
#     prob_a = np.zeros((self.Nr, self.Nr)) 
#     
#     # call the jitted function
#     LibraryBinaryNumeric_activity_correlations_brute_force_numba(
#         self.Ns, self.Nr, self.sens_mat,
#         self.substrate_probabilities, #< prob_s
#         np.empty(self.Nr, np.uint), #< ak
#         prob_a
#     )
#     return prob_a
# 
# 
# 
# numba_patcher.register_method(
#     'LibraryBinaryNumeric.activity_correlations_brute_force',
#     LibraryBinaryNumeric_activity_correlations_brute_force
# )

    
#===============================================================================
# MUTUAL INFORMATION BRUTE FORCE
#===============================================================================

            
@numba.jit(locals={'i': numba.int32, 'j': numba.int32},
           nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_fixed_numba(
        S_ni, hi, Jij, m, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Ns = S_ni.shape[1]
    indices = np.empty(m, np.uint)
    
    # initialize the mixture vector
    for i in range(m):
        indices[i] = i

    # iterate over all mixtures with a given number of substrates    
    running = True
    while running:
        # find the next iteration of the mixture
        for i in range(m - 1, -1, -1):
            if indices[i] + m != i + Ns:
                indices[i] += 1
                for j in range(i + 1, m):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            # set the last mixture
            for i in range(m):
                indices[i] = i
            running = False
        # `indices` now holds the indices of ones in the concentration vector

        # determine the resulting activity pattern 
        a_id = _activity_pattern_indices(indices, S_ni)
        
        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy_indices(indices, hi, Jij))

        prob_a[a_id] += pm
    
    # normalize prob_a and calculate its entropy    
    return _get_entropy_normalize(prob_a)
        
   

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_corr_numba(
        S_ni, hi, Jij, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Ns = S_ni.shape[1]
    ci = np.empty(Ns, np.uint)
    
    # iterate over all mixtures m
    for c in range(2**Ns):
        # extract the mixture from the single integer `c`
        for i in range(Ns):
            ci[i] = c % 2
            c //= 2
            
        # calculate the activity pattern id
        a_id = _activity_pattern(ci, S_ni)
        
        # calculate the probability of finding this mixture 
        pm = np.exp(-_mixture_energy(ci, hi, Jij))
        
        prob_a[a_id] += pm
    
    # normalize prob_a and calculate its entropy    
    return _get_entropy_normalize(prob_a)
       


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_brute_force_numba(
        S_ni, prob_s, prob_a):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Nr, Ns = S_ni.shape
    a_n = np.empty(Nr, np.uint)
    
    # iterate over all mixtures m
    for m in range(2**Ns):
        pm = 1     #< probability of finding this mixture
        a_n[:] = 0  #< activity pattern of this mixture
        # iterate through substrates in the mixture
        for i in range(Ns):
            r = m % 2
            m //= 2
            if r == 1:
                # substrate i is present
                pm *= prob_s[i]
                for a in range(Nr):
                    if S_ni[a, i] == 1:
                        a_n[a] = 1
            else:
                # substrate i is not present
                pm *= 1 - prob_s[i]
                
        # calculate the activity pattern id
        a_id, base = 0, 1
        for n in range(Nr):
            if a_n[n] == 1:
                a_id += base
            base *= 2
        
        prob_a[a_id] += pm
    
    # calculate the mutual information from the observed probabilities
    return _get_entropy(prob_a)
    


def LibraryBinaryNumeric_mutual_information_brute_force(self, ret_prob_activity=False):
    """ calculate the mutual information by constructing all possible
    mixtures """

    prob_a = np.zeros(2**self.Nr) 
    mixture_size = self.parameters['fixed_mixture_size']
    
    if mixture_size is not None:
        # call the jitted function for mixtures with fixed size
        MI = LibraryBinaryNumeric_mutual_information_brute_force_fixed_numba(
            self.sens_mat,
            self.commonness, self.correlations, #< hi, Jij
            int(mixture_size),
            prob_a
        )
    
    elif self.is_correlated_mixture:
        # call the jitted function for correlated mixtures
        MI = LibraryBinaryNumeric_mutual_information_brute_force_corr_numba(
            self.sens_mat,
            self.commonness, self.correlations, #< hi, Jij
            prob_a
        )
        
    else:
        # call the jitted function for uncorrelated mixtures
        MI = LibraryBinaryNumeric_mutual_information_brute_force_numba(
            self.sens_mat,
            self.substrate_probabilities, #< prob_s
            prob_a
        )
    
    if ret_prob_activity:
        return MI, prob_a
    else:
        return MI



numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_brute_force',
    LibraryBinaryNumeric_mutual_information_brute_force
)

    
#===============================================================================
# MUTUAL INFORMATION MONTE CARLO
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
        steps, S_ni, prob_s, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Nr, Ns = S_ni.shape
    a_n = np.empty(Nr, np.uint)
        
    # sample mixtures according to the probabilities of finding substrates
    for _ in range(steps):
        # choose a mixture vector according to substrate probabilities
        a_n[:] = 0  #< activity pattern of this mixture
        for i in range(Ns):
            if np.random.random() < prob_s[i]:
                # the substrate i is present in the mixture
                for n in range(Nr):
                    if S_ni[n, i] == 1:
                        # receptor a is activated by substrate i
                        a_n[n] = 1
        
        # calculate the activity pattern id
        a_id, base = 0, 1
        for n in range(Nr):
            if a_n[n] == 1:
                a_id += base
            base *= 2
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize prob_a and calculate its entropy    
    return _get_entropy_normalize(prob_a)
 
 

@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_metropolis_numba(
        steps, S_ni, hi, Jij, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Ns = S_ni.shape[1]
    ci = np.empty(Ns, np.uint8)
        
    # initialize the concentration vector
    for i in range(Ns):
        ci[i] = np.random.randint(2) #< set to either 0 or 1
    E_last = _mixture_energy(ci, hi, Jij)
    a_id = _activity_pattern(ci, S_ni)
        
    # sample mixtures according to the probabilities of finding substrates
    for _ in range(steps):
        # choose a new mixture based on the old one
        k = np.random.randint(Ns)
        ci[k] = 1 - ci[k]
        E_new = _mixture_energy(ci, hi, Jij)
        
        if E_new < E_last or np.random.random() < np.exp(E_last - E_new):
            # accept the new state
            E_last = E_new

            # calculate the activity pattern from this mixture vector
            a_id = _activity_pattern(ci, S_ni)
        
        else:
            # reject the new state and revert to the last one
            ci[k] = 1 - ci[k]
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize prob_a and calculate its entropy    
    return _get_entropy_normalize(prob_a)
 


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL) 
def LibraryBinaryNumeric_mutual_information_metropolis_swap_numba(
        steps, S_ni, hi, Jij, mixture_size, ind_0, ind_1, prob_a):
    """ calculate the mutual information using a monte carlo strategy. The
    number of steps is given by the model parameter 'monte_carlo_steps' """
    Ns = S_ni.shape[1]
    
    # find out how many zeros and ones there are => these numbers are fixed
    num_0 = Ns - mixture_size
    num_1 = mixture_size
    
    assert num_0 == len(ind_0)
    assert num_1 == len(ind_1)
    
    if num_0 == 0 or num_1 == 0:
        # there will be only a single mixture and the mutual information thus
        # vanishes trivially
        return 0

    # get the energy and activity pattern of the first mixture      
    E_last = _mixture_energy_indices(ind_1, hi, Jij)
    a_id = _activity_pattern_indices(ind_1, S_ni)
        
    # sample mixtures according to the probabilities of finding
    # substrates
    for _ in range(steps):
        # choose two substrates to swap. Here, we choose the want_0-th zero and
        # the want_1-th one in the vector and swap these two
        k0 = np.random.randint(num_0)
        k1 = np.random.randint(num_1)

        # switch the presence of the two substrates        
        ind_0[k0], ind_1[k1] = ind_1[k1], ind_0[k0] 
                
        # calculate the energy of the new mixture
        E_new = _mixture_energy_indices(ind_1, hi, Jij)
        
        if E_new < E_last or np.random.random() < np.exp(E_last - E_new):
            # accept the new mixture vector and save its energy
            E_last = E_new
                        
            # calculate the activity pattern id
            a_id = _activity_pattern_indices(ind_1, S_ni)
            
        else:
            # reject the new state and revert to the last one  -> we can also
            # reuse the calculations of the activity pattern from the last step
            ind_0[k0], ind_1[k1] = ind_1[k1], ind_0[k0] 
        
        # increment counter for this output
        prob_a[a_id] += 1
        
    # normalize prob_a and calculate its entropy    
    return _get_entropy_normalize(prob_a)

   

def LibraryBinaryNumeric_mutual_information_monte_carlo(self, ret_error=False,
                                                        ret_prob_activity=False,
                                                        bias_correction=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    prob_a = np.zeros(2**self.Nr)
    mixture_size = self.parameters['fixed_mixture_size']
 
    if mixture_size is not None:
        # use the version of the metropolis algorithm that keeps the number
        # of substrates in a mixture constant
        mixture_size = int(mixture_size)
        steps = self.get_steps('metropolis')
        
        # choose substrates that are present in the initial mixture
        ind_1 = np.random.choice(range(self.Ns), mixture_size, replace=False)
        ind_0 = np.array([i for i in range(self.Ns) if i not in ind_1])
        
        # call jitted function implementing swapping metropolis algorithm
        MI = LibraryBinaryNumeric_mutual_information_metropolis_swap_numba(
            steps, 
            self.sens_mat,
            self.commonness, self.correlations, #< hi, Jij
            mixture_size, ind_0, ind_1, prob_a
        )
    
    elif self.is_correlated_mixture:
        # mixture has correlations and we thus use a metropolis algorithm
        steps = self.get_steps('metropolis')
        
        # call jitted function implementing simple metropolis algorithm
        MI = LibraryBinaryNumeric_mutual_information_metropolis_numba(
            steps, 
            self.sens_mat,
            self.commonness, self.correlations, #< hi, Jij
            prob_a
        )
    
    else:
        # simple case without correlations and unconstrained number of ligands
        steps = self.get_steps('monte_carlo')
        
        # call jitted function implementing simple monte carlo algorithm
        MI = LibraryBinaryNumeric_mutual_information_monte_carlo_numba(
            steps, 
            self.sens_mat,
            self.substrate_probabilities, #< prob_s
            prob_a
        )
        
    if bias_correction:
        # add entropy bias correction, MLE of [Paninski2003]
        MI += (np.count_nonzero(prob_a) - 1)/(2*steps)

    if ret_error:
        # estimate the error of the mutual information calculation
        MI_err = sum(np.abs(1/np.log(2) + np.log2(pa)) * pa
                     for pa in prob_a if pa != 0) / np.sqrt(steps)

        if ret_prob_activity:
            return MI, MI_err, prob_a
        else:
            return MI, MI_err

    else:    
        # do not estimate the error of the mutual information calculation
        if ret_prob_activity:
            return MI, prob_a
        else:
            return MI



numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_monte_carlo',
    LibraryBinaryNumeric_mutual_information_monte_carlo,
    check_return_value_approx
)

    
#===============================================================================
# MUTUAL INFORMATION ESTIMATION
#===============================================================================


@numba.jit(nopython=NUMBA_NOPYTHON, nogil=NUMBA_NOGIL)
def _mutual_information_from_q(q_n, q_nm):
    """ estimates the mutual information from q_n and q_nm """
    Nr = len(q_n)
    LN2 = np.log(2) #< compile-time constant
    
    MI = 0
    for n in range(Nr):
        if 0 < q_n[n] < 1:
            #MI -= 0.5/LN2 * (1 - 2*q_n[n])**2
            MI -= q_n[n]*np.log2(q_n[n]) + (1 - q_n[n])*np.log2(1 - q_n[n]) 
        for m in range(n):
            MI -= 8/LN2 * q_nm[n, m]**2
    return MI 


@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_approx_numba(
        S_ni, prob_s):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Nr, Ns = S_ni.shape
    q_n = np.zeros(Nr)
    q_nm = np.zeros((Nr, Nr))
    
    # iterate over all receptors to estimate crosstalk
    for n in range(Nr):
        # evaluate the probability that a receptor gets activated by ligand i
        for i in range(Ns):
            if S_ni[n, i] == 1:
                q_n[n] += prob_s[i]
                
                # calculate crosstalk with other receptors
                for m in range(Nr):
                    if n != m and S_ni[m, i] == 1:
                        q_nm[n, m] += prob_s[i]

    # estimate mutual information
    return _mutual_information_from_q(q_n, q_nm)
    
    
@numba.jit(locals={'i_count': numba.int32}, nopython=NUMBA_NOPYTHON,
           nogil=NUMBA_NOGIL)
def LibraryBinaryNumeric_mutual_information_estimate_numba(
        S_ni, prob_s):
    """ calculate the mutual information by constructing all possible
    mixtures """
    Nr, Ns = S_ni.shape
    q_n = np.empty(Nr)            
    q_nm = np.zeros((Nr, Nr))
    ids = np.empty(Ns, np.int32)

    # iterate over all receptors to determine q_n and q_nm
    for n in range(Nr):
        # evaluate the direct
        i_count = 0 #< number of substrates that excite receptor n
        prod = 1    #< product important for calculating the probabilities
        for i in range(Ns):
            if S_ni[n, i] == 1:
                prod *= 1 - prob_s[i]
                ids[i_count] = i
                i_count += 1
        q_n[n] = 1 - prod

        # calculate crosstalk
        for m in range(Nr):
            if n != m:
                prod = 1
                for k in range(i_count):
                    if S_ni[m, ids[k]] == 1:
                        prod *= 1 - prob_s[ids[k]]
                q_nm[n, m] = 1 - prod

    # estimate mutual information
    return _mutual_information_from_q(q_n, q_nm)



def LibraryBinaryNumeric_mutual_information_estimate(self, approx_prob=False):
    """ calculate the mutual information by constructing all possible
    mixtures """
    if self.is_correlated_mixture:
        logging.warning('Numba code not implemented for correlated mixtures. '
                        'Falling back to pure-python method.')
        this = LibraryBinaryNumeric_mutual_information_estimate
        return this._python_function(self, approx_prob)

    if approx_prob:
        # call the jitted function that uses approximate probabilities
        MI = LibraryBinaryNumeric_mutual_information_estimate_approx_numba(
            self.sens_mat,
            self.substrate_probabilities
        )

    else:    
        # call the jitted function that uses exact probabilities
        MI = LibraryBinaryNumeric_mutual_information_estimate_numba(
            self.sens_mat,
            self.substrate_probabilities
        )
    
    return MI



# Temporarily disable the numba method since we change it frequently
numba_patcher.register_method(
    'LibraryBinaryNumeric.mutual_information_estimate',
    LibraryBinaryNumeric_mutual_information_estimate,
    test_function=check_return_value_exact
)

