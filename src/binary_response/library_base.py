'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import copy
import functools
import logging
import multiprocessing as mp

import numpy as np
from scipy import special
from six import string_types

from utils.math import xlog2x
from utils.math.stats import StatisticsAccumulator
from utils.numba.tools import random_seed


LN2 = np.log(2)

# define vectorize function for double results to use as a decorator
vectorize_double = functools.partial(np.vectorize, otypes=[np.double])
# TODO: try using numba vectorize to speed up 


class LibraryBase(object):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'ensemble_average_num': 32,      #< repetitions for ensemble average
        'multiprocessing_cores': 'auto', #< number of cores to use
         # how to initialize the state
        'initialize_state': {'default': 'auto'},
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        self.Ns = num_substrates
        self.Nr = num_receptors
        
        # initialize parameters with default ones from all parent classes
        self.parameters = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                # we need to make a deep copy to copy nested dictionaries
                self.parameters.update(copy.deepcopy(cls.parameters_default))
                
        # update parameters with the supplied ones
        if parameters is not None:

            # remove old definitions of `initialize_state` 
            if isinstance(parameters.get('initialize_state'), string_types):
                logging.warning('Initialized model with old `initialize_state` '
                                'that is a string.')
                parameters.pop('initialize_state')
            
            self.parameters.update(parameters)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        return ['Ns=%d' % self.Ns, 'Nr=%d' % self.Nr]


    def __repr__(self):
        params = ', '.join(self.repr_params)
        return '%s(%s)' % (self.__class__.__name__, params)


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        return {'num_substrates': self.Ns,
                'num_receptors': self.Nr,
                'parameters': self.parameters}

    
    def copy(self):
        """ returns a copy of the current object """
        return self.__class__(**self.init_arguments)


    @classmethod
    def from_other(cls, other, **kwargs):
        """ creates an instance of this class by using parameters from another
        instance """
        # create object with parameters from other object
        init_arguments = other.init_arguments
        init_arguments.update(kwargs)
        return cls(**init_arguments)


    @classmethod
    def get_random_arguments(cls, num_substrates=None, num_receptors=None,
                             parameters=None):
        """ create random arguments for creating test instances """
        if num_substrates is None:
            num_substrates = np.random.randint(3, 6)
        if num_receptors is None:
            num_receptors =  np.random.randint(2, 4)
        if parameters is None:
            parameters = {}
        
        # return the dictionary
        return {'num_substrates': num_substrates,
                'num_receptors': num_receptors,
                'parameters': parameters}


    @classmethod
    def create_test_instance(cls, **kwargs):
        """ creates a test instance used for consistency tests """
        return cls(**cls.get_random_arguments(**kwargs))
  
  
    @property
    def mutual_information_max(self):
        """ returns an upper bound to the mutual information """
        return self.Nr
  
            
    def get_number_of_cores(self):
        """ returns the number of cores to use in multiprocessing """
        multiprocessing_cores = self.parameters['multiprocessing_cores']
        if multiprocessing_cores == 'auto':
            return mp.cpu_count()
        else:
            return multiprocessing_cores
            
            
    def ensemble_average(self, method, avg_num=None, multiprocessing=False, 
                         ret_all=False, args=None, initialize_state='ensemble'):
        """ calculate an ensemble average of the result of the `method` of
        multiple different receptor libraries.
        
        `avg_num` is the number of receptor libraries that we average over
        `multiprocessing` determines whether the calculations run in parallel.
            The number of cores that are used can be specified by the class 
            parameter `multiprocessing_cores`.
        `ret_all` is a flag determining whether all results are returned as a
            list of length `avg_num` or whether the ensemble mean and the
            ensemble standard deviation are returned.
        `args` is a dictionary with additional arguments supplied to `method`
        `initialize_state` is a flag that determines how the receptor libraries
            are initialized. This influences which parameters are the same
            across the ensemble.
        """
        
        if avg_num is None:
            avg_num = self.parameters['ensemble_average_num']
        if args is None:
            args = {}
        
        if multiprocessing and avg_num > 1:
            
            init_arguments = self.init_arguments
            
            # set the initialization procedure to ensemble, such that new
            # realizations are chosen at each iteration
            if initialize_state is not None:
                init_state = init_arguments['parameters']['initialize_state']
                init_state['default'] = initialize_state
            
            # run the calculations in multiple processes
            arguments = (self.__class__, init_arguments, method, args)
            pool = mp.Pool(processes=self.get_number_of_cores())
            results = pool.map(_ensemble_average_job, [arguments] * avg_num)
            
            # Apparently, multiprocessing sometimes opens too many files if
            # processes are launched to quickly and the garbage collector cannot
            # keep up. We thus explicitly terminate the pool here.
            pool.terminate()
            
        else:
            # run the calculations in this process
            cls = self.__class__
            results = [getattr(cls(**self.init_arguments), method)(**args)
                       for _ in range(avg_num)]
    
        # collect the results and calculate the statistics
        if ret_all:
            return results
        else:
            return self._result_statistics(results)
        
        
    def _result_statistics(self, results):
        """ returns the means and the standard deviation of the `results` """
        
        if isinstance(results[0], dict):
            # average all keys of the dict individually
            acc_list = {key: StatisticsAccumulator() for key in results[0]}
            
            # iterate through all results and build the statistics
            for result in results:
                for key, value in result.items():
                    acc_list[key].add(value)

            # return the statistics
            means = {key: stats.mean for key, stats in acc_list.items()}
            stds = {key: stats.std for key, stats in acc_list.items()}
            return means, stds
        
        # determine the format of the result
        try:
            shapes = set([v.shape for v in results[0]])
        except (TypeError, AttributeError):
            # results[0] was either not a list or its items are not numpy arrays
            # => assume that individual results are numbers or arrays
            handle_as_array = True
        else:
            # results[0] is a list of numpy arrays
            handle_as_array = (len(shapes) == 1) 

        # calculate the statistics with the determined method 
        if handle_as_array:
            # handle result as one array
            results = np.array(results)
            return results.mean(axis=0), results.std(axis=0)
        
        else:
            # handle list items separately
            acc_list = [StatisticsAccumulator() for _ in range(len(results[0]))]
            
            # iterate through all results and build the statistics
            for result in results:
                for k, dataset in enumerate(result):
                    acc_list[k].add(dataset)
                    
            # return the statistics
            means = [stats.mean for stats in acc_list]
            stds = [stats.std for stats in acc_list]
            return means, stds


    def ctot_statistics(self, **kwargs):
        """ returns the statistics for the total concentration. All arguments
        are passed to the call to `self.concentration_statistics` """
        # get the statistics of the individual substrates
        c_stats = self.concentration_statistics(**kwargs)
        
        # calculate the statistics of their sum
        ctot_mean = c_stats['mean'].sum()
        if c_stats.get('cov_is_diagonal', False):
            ctot_var = c_stats['var'].sum()
        else:
            ctot_var = c_stats['cov'].sum()
        
        return {'mean': ctot_mean, 'std': np.sqrt(ctot_var), 'var': ctot_var}
    
            
    def _estimate_qn_from_en(self, en_stats, excitation_model='default'):
        """ estimates probability q_n that a receptor is activated by a mixture
        based on the statistics of the excitations en """

        if excitation_model == 'default':
            excitation_model = 'log-normal'

        if 'gauss' in excitation_model:
            if 'approx' in excitation_model:
                # estimate from a simple expression, which was obtained from
                # expanding the expression from the Gaussian
                q_n = _estimate_qn_from_en_gaussian_approx(en_stats['mean'],
                                                           en_stats['var'])
            else:
                # estimate from a gaussian distribution
                q_n = _estimate_qn_from_en_gaussian(en_stats['mean'],
                                                    en_stats['var'])

        elif 'log-normal' in excitation_model or 'lognorm' in excitation_model:
            if 'approx' in excitation_model:
                # estimate from a simple expression, which was obtained from
                # expanding the expression from the log-normal
                q_n = _estimate_qn_from_en_lognorm_approx(en_stats['mean'],
                                                          en_stats['var'])
            else:
                # estimate from a log-normal distribution
                q_n = _estimate_qn_from_en_lognorm(en_stats['mean'],
                                                   en_stats['var'])

        elif 'trunc-normal' in excitation_model:
            # use a truncated normal distribution with estimated _mean and
            # variance
            q_n = _estimate_qn_from_en_truncnorm(en_stats['mean'],
                                                 en_stats['var'])

        elif 'gamma' in excitation_model:
            # use a Gamma distribution with estimated _mean and variance
            q_n = _estimate_qn_from_en_gamma(en_stats['mean'], en_stats['var'])

        else:
            raise ValueError('Unknown excitation model `%s`' % excitation_model)
            
        return q_n
   
    
    def _estimate_qnm_from_en(self, en_stats):
        """ estimates crosstalk q_nm based on the statistics of the excitations
        en """
        en_cov = en_stats['cov']
        
        # calculate the correlation coefficient
        if np.isscalar(en_cov):
            # scalar case
            en_var = en_stats['var']
            if np.isclose(en_var, 0):
                rho = 0
            else:
                rho = en_cov / en_var
            
        else:
            # matrix case
            en_std = en_stats['std'] 
            with np.errstate(divide='ignore', invalid='ignore'):
                rho = np.divide(en_cov, np.outer(en_std, en_std))
    
            # replace values that are nan with zero. This might not be exact,
            # but only occurs in corner cases that are not interesting to us  
            rho[np.isnan(rho)] = 0
            
        # estimate the crosstalk
        q_nm = rho / (2*np.pi)
            
        return q_nm
    
            
    def _estimate_MI_from_q_values(self, q_n, q_nm, method='default'):
        """ estimate the mutual information from given probabilities
        All approximations to the mutual information are based on the
        approximations given in 
            V. Sessak and R. Monasson, J Phys A, 42, 055001 (2009)
            
        `method` selects one of the following approximations:
            [`expansion`, `hybrid`, `polynom`] 
        """ 
        
        if method == 'default':
            method = 'hybrid'
        
        if method == 'expansion':
            # use the formula from the paper directly
            MI = -np.sum(xlog2x(q_n) + xlog2x(1 - q_n))
        
            # calculate the crosstalk
            q_n2 = q_n**2 - q_n
            with np.errstate(divide='ignore', invalid='ignore'):
                q_nm_scaled = q_nm**2 / np.outer(q_n2, q_n2)
            
            # replace values that are not finite with zero. This might not be
            # exact, but only occurs in cases that are not interesting to us  
            q_nm_scaled[~np.isfinite(q_nm_scaled)] = 0
            
            MI -= 0.5/LN2 * np.sum(np.triu(q_nm_scaled, 1))
        
        elif method == 'hybrid':
            # use the exact first term, but expand the second
            MI = -np.sum(xlog2x(q_n) + xlog2x(1 - q_n))
        
            # calculate the crosstalk
            MI -= 8/LN2 * np.sum(np.triu(q_nm, 1)**2)

        elif method == 'polynom':
            # use the quadratic approximation of the mutual information
            MI = self.Nr - 0.5/LN2 * np.sum((2*q_n - 1)**2)
            # calculate the crosstalk
            MI -= 8/LN2 * np.sum(np.triu(q_nm, 1)**2)
            
        else:
            raise ValueError('Unknown method `%s` for calculating MI' % method)
            
        return MI
    
        
    def _estimate_MI_from_q_stats(self, q_n, q_nm, q_n_var=0, q_nm_var=0,
                                  method='default'):
        """ estimate the mutual information from given probabilities
        All approximations to the mutual information are based on the
        approximations given in 
            V. Sessak and R. Monasson, J Phys A, 42, 055001 (2009)
            
        `method` selects one of the following approximations:
            [`expansion`, `hybrid`, `polynom`] 
            
        Independent of the method chosen, the variances are always only used
        in a quadratic approximation.
        """
        Nr = self.Nr
        
        if method == 'default':
            method = 'hybrid'
        
        if method == 'expansion':
            # use the formula from the paper directly

            # use exact expression for the entropy of uncorrelated receptors             
            MI = -Nr * (xlog2x(q_n) + xlog2x(1 - q_n))

            # add the effect of crosstalk
            if 0 < q_n < 1: 
                MI -= 0.5/LN2 * Nr*(Nr - 1)/2 * q_nm**2 / (q_n**2 - q_n)**2
        
        elif method == 'hybrid':
            # use the exact first term, but expand the second
            
            # use exact expression for the entropy of uncorrelated receptors             
            MI = -Nr * (xlog2x(q_n) + xlog2x(1 - q_n))

            # add the effect of crosstalk
            MI -= 8/LN2 * Nr*(Nr - 1)/2 * q_nm**2

        elif method == 'polynom':
            # use the quadratic approximation of the mutual information
            MI = Nr - 0.5/LN2 * Nr * (2*q_n - 1)**2
            # add the effect of crosstalk
            MI -= 8/LN2 * Nr*(Nr - 1)/2 * q_nm**2
            
        else:
            raise ValueError('Unknown method `%s` for calculating MI' % method)
                   
        # add the effect of the variance         
        if q_n_var != 0:
            MI -= 2/LN2 * Nr * q_n_var
        if q_nm_var != 0:
            MI -= 4/LN2 * Nr*(Nr - 1)/2 * q_nm_var
                            
        return MI
    
        
    def _estimate_MI_from_r_values(self, r_n, r_nm, method='default'):
        """ estimate the mutual information from given probabilities """
        # calculate the crosstalk
        q_nm = r_nm - np.outer(r_n, r_n)
        return self._estimate_MI_from_q_values(r_n, q_nm, method)
      
        
    def _estimate_MI_from_r_stats(self, r_n, r_nm, r_n_var=0, r_nm_var=0,
                                  method='default'):
        """ estimate the mutual information from given probabilities """
        if r_nm_var != 0:
            raise NotImplementedError('Correlation calculations are not tested.')
        # calculate the crosstalk
        q_nm = r_nm - r_n**2 - r_n_var
        q_nm_var = r_nm_var + 4*r_n**2*r_n_var + 2*r_n_var**2
        return self._estimate_MI_from_q_stats(r_n, q_nm, r_n_var, q_nm_var,
                                              method=method)
    


@vectorize_double
def _estimate_qn_from_en_lognorm(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n assuming an underlying
    log-normal distribution for e_n """
    if en_mean == 0:
        q_n = 0.
    elif np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:
        en_cv2 = en_var / en_mean**2
        enum = np.log(np.sqrt(1 + en_cv2) / en_mean)
        denom = np.sqrt(2*np.log1p(en_cv2))
        q_n = 0.5 * special.erfc(enum/denom)
        
    return q_n
       
               

@vectorize_double
def _estimate_qn_from_en_lognorm_approx(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n using an approximation """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:                
        q_n = (0.5
               + (en_mean - 1) / np.sqrt(2*np.pi*en_var)
               + (5*en_mean - 7) * np.sqrt(en_var/(32*np.pi))
               )
        # here, the last term comes from an expansion of the log-normal approx.

    return np.clip(q_n, 0, 1)



@vectorize_double
def _estimate_qn_from_en_gaussian(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n assuming an underlying
    normal distribution for e_n """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:
        q_n = 0.5 * special.erfc((1 - en_mean)/np.sqrt(2 * en_var))
        
    return q_n
           
               

@vectorize_double
def _estimate_qn_from_en_gaussian_approx(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n using an approximation """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:                
        q_n = 0.5 + (en_mean - 1) / np.sqrt(2*np.pi*en_var)

    return np.clip(q_n, 0, 1)



@vectorize_double
def _estimate_qn_from_en_truncnorm(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n assuming an underlying
    truncated normal distribution for e_n """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:     
        fac = np.sqrt(2) * en_var
        enum = 1 + special.erf((en_mean - 1)/fac)
        denom = 1 + special.erf(en_mean/fac)
        q_n = enum / denom

    return np.clip(q_n, 0, 1)



@vectorize_double
def _estimate_qn_from_en_gamma(en_mean, en_var):
    """ estimates probability q_n that a receptor is activated by a mixture
    based on the statistics of the excitations e_n assuming an underlying
    Gamma distribution for e_n """
    if np.isclose(en_var, 0):
        q_n = np.double(en_mean >= 1)
    else:     
        b = en_mean / en_var
        q_n = special.gammaincc(en_mean*b, b)

    return np.clip(q_n, 0, 1)



def _ensemble_average_job(args):
    """ helper function for calculating ensemble averages using
    multiprocessing """
    # We have to initialize the random number generator for each process
    # because we would have the same random sequence for all processes
    # otherwise.
    random_seed()
    
    # create the object ...
    obj = args[0](**args[1])
    # ... and evaluate the requested method
    if len(args) > 2: 
        return getattr(obj, args[2])(**args[3])
    else:
        return getattr(obj, args[2])()

    
    