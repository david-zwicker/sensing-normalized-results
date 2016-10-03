'''
Created on Mar 31, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import integrate

from .lib_spr_base import LibrarySparseBase
from utils.math.distributions import (lognorm_mean, loguniform_mean,
                                      DeterministicDistribution)



class LibrarySparseTheoryBase(LibrarySparseBase):
    """ base class for theoretical libraries for sparse mixtures """
    
    
    def sensitivity_stats(self):
        """ returns the statistics of the sensitivity matrix """
        raise NotImplementedError("Needs to be implemented by subclass")
    
    
    def excitation_statistics(self):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        # get statistics of the total concentration c_tot = \sum_i c_i
        c_stats = self.concentration_statistics()
        ctot_mean = c_stats['mean'].sum()
        ctot_var = c_stats['var'].sum()
        c2_mean = c_stats['mean']**2 + c_stats['var']
        c2_mean_sum = c2_mean.sum()
        
        # get statistics of the sensitivities S_ni
        S_stats = self.sensitivity_stats()
        S_mean = S_stats['mean']
        
        # calculate statistics of the sum e_n = \sum_i S_ni * c_i        
        en_mean = S_mean * ctot_mean
        en_var  = S_mean**2 * ctot_var + S_stats['var'] * c2_mean_sum
        enm_cov = S_mean**2 * ctot_var + S_stats['cov'] * c2_mean_sum

        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
        
    
    def receptor_activity(self, ret_correlations=False,
                          excitation_model='default', clip=True):
        """ return the probability with which a single receptor is activated 
        by typical mixtures """
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                            excitation_model=excitation_model,
                                            clip=False)
        
        r_n = q_n
        r_nm = q_n**2 + q_nm
        
        if clip:
            r_n = np.clip(r_n, 0, 1)
            r_nm = np.clip(r_nm, 0, 1)
        
        if ret_correlations:
            return r_n, r_nm
        else:
            return r_n
        
        
    def receptor_crosstalk(self, ret_receptor_activity=False,
                           excitation_model='default', clip=True):
        """ calculates the average activity of the receptor as a response to 
        single ligands. """
        en_stats = self.excitation_statistics()
        
        if en_stats['mean'] == 0:
            q_n, q_nm = 0, 0
            
        else:
            q_n = self._estimate_qn_from_en(en_stats, excitation_model)
            q_nm = self._estimate_qnm_from_en(en_stats)
                
            if clip:
                q_n = np.clip(q_n, 0, 1)
                q_nm = np.clip(q_nm, 0, 1)

        if ret_receptor_activity:
            return q_n, q_nm
        else:
            return q_nm


    def mutual_information(self, excitation_model='default',
                           mutual_information_method='default', clip=False):
        """ calculates the typical mutual information """
        # get receptor activity probabilities
        q_n, q_nm = self.receptor_crosstalk(ret_receptor_activity=True,
                                            excitation_model=excitation_model)
        
        # estimate mutual information from this
        MI = self._estimate_MI_from_q_stats(
                                    q_n, q_nm, method=mutual_information_method)
        
        if clip:
            return np.clip(MI, 0, self.Nr)
        else:
            return MI

        
    def set_optimal_parameters(self, **kwargs):
        """ adapts the parameters of this library to be close to optimal """
        params = self.get_optimal_library()
        del params['distribution']
        self.__dict__.update(params)



class LibrarySparseBinary(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries. The only
    parameters that characterizes this library is the density of entries and
    their magnitude """


    def __init__(self, num_substrates, num_receptors,
                 mean_sensitivity=1, parameters=None, **kwargs):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the fraction `density` of
        substrates a single receptor responds to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix """
        super(LibrarySparseBinary, self).__init__(num_substrates,
                                                  num_receptors, parameters)

        self.mean_sensitivity = mean_sensitivity
        
        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            S_mean2 = mean_sensitivity**2
            self.density = S_mean2 / (S_mean2 + standard_deviation**2)
        elif 'density' in kwargs:
            self.density = kwargs.pop('density')
        else:
            standard_deviation = 1
            S_mean2 = mean_sensitivity**2
            self.density = S_mean2 / (S_mean2 + standard_deviation**2)

        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs)) 


    @property
    def standard_deviation(self):
        """ returns the standard deviation of the sensitivity matrix """
        return self.mean_sensitivity * np.sqrt(1/self.density - 1)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseBinary, self).repr_params
        params.append('xi=%g' % self.density)
        params.append('<S>=%g' % self.mean_sensitivity)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseBinary, self).init_arguments
        args['density'] = self.density
        args['mean_sensitivity'] = self.mean_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseBinary, cls).get_random_arguments(**kwargs)
        args['density'] = kwargs.get('density', np.random.random())
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        return args


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        var = S0**2 * (1/self.density - 1)
        return {'mean': S0, 'var': var, 'std': np.sqrt(var), 'cov': 0}


    def density_optimal(self, assume_homogeneous=False):
        """ return the estimated optimal activity fraction for the simple case
        where all h are the same. The estimate relies on an approximation that
        all receptors are independent and is thus independent of the number of 
        receptors. The estimate is thus only good in the limit of low Nr.
        
        If `assume_homogeneous` is True, the calculation is also done in the
            case of heterogeneous mixtures, where the probability of the
            homogeneous system with the same average number of substrates is
            used instead.
        """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        if assume_homogeneous:
            # calculate the idealized substrate probability
            m_mean = self.mixture_size_statistics()['mean']
            p0 = m_mean / self.Ns
             
        else:
            # check whether the mixtures are all homogeneous
            if len(np.unique(self.commonness)) > 1:
                raise RuntimeError('The estimate only works for homogeneous '
                                   'mixtures so far.')
            p0 = self.substrate_probabilities.mean()
            
        # calculate the fraction for the homogeneous case
        return (1 - 2**(-1/self.Ns))/p0
    
    
    def get_optimal_library(self):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        m = self.mixture_size_statistics()['mean']
        d = self.concentration_statistics()['mean'].mean()
        density = self.density_optimal()
        S0 = 1 / (m*d*density + d*np.log(2)) / density
        return {'distribution': 'binary',
                'mean_sensitivity': S0, 'density': density}



class LibrarySparseLogNormal(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries drawn from a
    log-normal distribution """


    def __init__(self, num_substrates, num_receptors, mean_sensitivity=1,
                 correlation=0, parameters=None, **kwargs):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, and the typical sensitivity
        or magnitude S0 of the sensitivity matrix.
        The width of the distribution is either set by the parameter `width` or
        by setting the `standard_deviation`.
        """
        super(LibrarySparseLogNormal, self).__init__(num_substrates,
                                                     num_receptors, parameters)
        
        self.mean_sensitivity = mean_sensitivity
        self.correlation = correlation
        
        if 'variance' in kwargs:
            kwargs['standard_deviation'] = np.sqrt(kwargs.pop('variance'))

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
    def variance(self):
        """ return the variance of the distribution """
        return self.mean_sensitivity**2 * (np.exp(self.width**2) - 1)
            

    @property
    def standard_deviation(self):
        """ return the standard deviation of the distribution """
        return self.mean_sensitivity * np.sqrt((np.exp(self.width**2) - 1))
            

    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseLogNormal, self).repr_params
        params.append('<S>=%g' % self.mean_sensitivity)
        params.append('width=%g' % self.width)
        params.append('correlation=%g' % self.correlation)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseLogNormal, self).init_arguments
        args['mean_sensitivity'] = self.mean_sensitivity
        args['width'] = self.width
        args['correlation'] = self.correlation
        return args


    @classmethod
    def get_random_arguments(cls, mean_sensitivity=None, width=None, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseLogNormal, cls).get_random_arguments(**kwargs)

        if width is None:
            args['width'] = np.random.random() + 0.5
        else:
            args['width'] = width
        if mean_sensitivity is None:
            args['mean_sensitivity'] = np.random.random() + 1
        else: 
            args['mean_sensitivity'] = mean_sensitivity
        return args


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


    def get_optimal_parameters(self, fixed_parameter='width'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        ci_stats = self.concentration_statistics()
        ci2_sum = np.sum(ci_stats['mean']**2)

        ctot_stats = self.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']
        ctot_cv2 = ctot_var / ctot_mean**2
        
        if fixed_parameter == 'width':
            # keep the width parameter fixed and determine the others 
            width_opt = self.width
            
            ctot_stats2 = (ctot_var + ci2_sum) / ctot_mean**2
            braket = 1 + ctot_cv2 + ctot_stats2 * (np.exp(width_opt**2) - 1)
            S0_opt = np.sqrt(braket) / ctot_mean
            std_opt = S0_opt * np.sqrt(np.exp(width_opt**2) - 1)
            
        elif fixed_parameter == 'S0':
            # keep the typical sensitivity fixed and determine the other params
            S0_opt = self.mean_sensitivity
            
            braket = (S0_opt * ctot_mean)**2
            width_term = braket - 1 - ctot_cv2
            width_braket = width_term * ctot_mean**2 / (ctot_var + ci2_sum)
            
            if width_braket >= 0:
                width_opt = np.sqrt(np.log(width_braket + 1))
                std_opt = self.mean_sensitivity * np.sqrt(width_braket)
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
    
    
    def get_optimal_library(self, fixed_parameter='width'):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices.
            `fixed_parameter` determines which parameter is kept fixed during
                the optimization procedure
        """
        library_opt = self.get_optimal_parameters(fixed_parameter)
        return {'distribution': 'log_normal', 'width': library_opt['width'],
                'mean_sensitivity': library_opt['mean_sensitivity'],
                'correlation': 0}
                
        
    def activity_distance_mixtures(self, mixture_size, mixture_overlap=0,
                                   concentration=None):
        """ calculates the expected Hamming distance between the activation
        pattern of two mixtures with `mixture_size` ligands of equal 
        concentration `concentration`. `mixture_overlap` denotes the number of
        ligands that are the same in the two mixtures """
        if mixture_overlap == mixture_size:
            return 0
    
        if concentration is None:
            concentration = self.c_means.mean()
    
        # load sped up function from numba code
        try:
            from utils.numba_tools import lognorm_pdf, lognorm_cdf
        except ImportError:
            raise ImportError("Calculating the mixture distance is currently "
                              "only supported if numba is available.")

        # introduce some abbreviations    
        c = concentration
        sB = mixture_overlap
        s = mixture_size
        S_stats = self.sensitivity_stats() 
    
        if sB == 0:
            # probability that one excitation is below threshold
            p_1below = lognorm_cdf(1/c, s*S_stats['mean'], s*S_stats['var'])
            # probability that both excitations are below threshold
            p_2below = p_1below**2
            # probability that both eps_L and eps_R bring it above threshold
            p_2above = (1 - p_1below)**2
            p_same = p_2below + p_2above
    
        else:
            # sB > 0:
            sD = s - sB
            def integrand(eps):
                """ probability that either one of the excitations caused by the
                different ligand brings the total excitation above threshold   
                given a certain excitation eps caused by the same ligands """
                p_below_thresh = lognorm_cdf(eps, sD*S_stats['mean'],
                                             sD*S_stats['var'])
                p_eps = lognorm_pdf(1/c - eps, sB*S_stats['mean'],
                                    sB*S_stats['var'])
                return p_below_thresh * (1 - p_below_thresh) * p_eps
            
            p_1diff = integrate.quad(integrand, 0, 1/c)
            p_same = 1 - 2*p_1diff[0]
        
        return self.Nr*(1 - p_same)
        
        

class LibrarySparseLogUniform(LibrarySparseTheoryBase):
    """ represents a single receptor library with random entries drawn from a
    log-uniform distribution """


    def __init__(self, num_substrates, num_receptors, width=1,
                 mean_sensitivity=1, parameters=None):
        """ initialize the receptor library by setting the number of receptors,
        the number of substrates it can respond to, the width of the
        distribution `width`, and the typical sensitivity or magnitude S0 of the
        sensitivity matrix """
        super(LibrarySparseLogUniform, self).__init__(num_substrates,
                                                      num_receptors, parameters)
        self.width = width
        self.mean_sensitivity = mean_sensitivity


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseLogUniform, self).repr_params
        params.append('width=%g' % self.width)
        params.append('<S>=%g' % self.mean_sensitivity)
        return params


    @property
    def init_arguments(self):
        """ return the parameters of the model that can be used to reconstruct
        it by calling the __init__ method with these arguments """
        args = super(LibrarySparseLogUniform, self).init_arguments
        args['width'] = self.width
        args['mean_sensitivity'] = self.mean_sensitivity
        return args


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibrarySparseLogUniform, cls).get_random_arguments(**kwargs)
        args['width'] = kwargs.get('width', np.random.random() + 0.1)
        S0 = np.random.random() + 0.5
        args['mean_sensitivity'] = kwargs.get('mean_sensitivity', S0)
        return args


    @property
    def sensitivity_distribution(self):
        """ returns the sensitivity distribution """
        if self.width == 0:
            return DeterministicDistribution(self.mean_sensitivity)
        else:
            return loguniform_mean(self.mean_sensitivity, np.exp(self.width))


    def sensitivity_stats(self):
        """ returns statistics of the sensitivity distribution """
        S0 = self.mean_sensitivity
        width = self.width
        
        if width == 0:
            var = 0
        else:
            exp_s2 = np.exp(width)**2
            var = S0**2 * (1 - exp_s2 + (1 + exp_s2)*width)/(exp_s2 - 1)
            
        return {'mean': S0, 'var': var, 'cov': 0}
    

    def get_optimal_library(self, width_opt=2):
        """ returns an estimate for the optimal parameters for the random
        interaction matrices """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        ctot_stats = self.ctot_statistics()
        ctot_mean = ctot_stats['mean']
        ctot_var = ctot_stats['var']

        if self.width == 0:
            term = 1 
        else:
            exp_s2 = np.exp(self.width)**2
            term = (exp_s2 + 1) * self.width / (exp_s2 - 1)
        S0_opt = np.sqrt(1 + ctot_var/ctot_mean**2 * term) / ctot_mean 
        
        return {'distribution': 'log_normal',
                'mean_sensitivity': S0_opt, 'width': width_opt}
        
        
        