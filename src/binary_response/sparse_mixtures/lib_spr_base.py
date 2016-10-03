'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging

import numpy as np
from scipy import stats

from ..binary_mixtures.lib_bin_base import LibraryBinaryBase
from utils.math.distributions import lognorm_mean_var, DeterministicDistribution



class LibrarySparseBase(LibraryBinaryBase):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `set_commonness`.  
    """
    
    # supported concentration distributions 
    concentration_distributions = ['exponential', 'log-normal', 'bernoulli']
    

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'c_distribution': 'exponential', 
        'c_mean_vector': None,     #< chosen substrate c_means
        'c_mean_parameters': None, #< parameters for substrate concentration
        'c_var_vector': None,
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibrarySparseBase, self).__init__(num_substrates, num_receptors,
                                                parameters)

        c_dists = self.concentration_distributions
        if self.parameters['c_distribution'] not in c_dists:
            raise ValueError('Concentration distribution `%s` is not in '
                             'supported distributions: %s.'
                             % (self.parameters['c_distribution'], c_dists))

        # determine how to initialize the variables
        init_state = self.parameters['initialize_state']
        
        # determine how to initialize the c_means
        init_c_mean = init_state.get('c_mean', init_state['default'])
        if init_c_mean  == 'auto':
            if self.parameters['c_mean_parameters'] is None:
                init_c_mean = 'exact'
            else:
                init_c_mean = 'ensemble'

        # initialize the c_means with the chosen method            
        if init_c_mean is None:
            self.c_means = None
            
        elif init_c_mean  == 'exact':
            logging.debug('Initialize with given c_means')
            self.c_means = self.parameters['c_mean_vector']
            
        elif init_c_mean == 'ensemble':
            conc_params = self.parameters['c_mean_parameters']
            if conc_params:
                logging.debug('Choose c_means from given parameters')
                self.choose_concentrations(**conc_params)
            else:
                logging.warning('Requested to set c_means from parameters, '
                                'but parameters were not supplied.')
                self.c_means = None
                    
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_c_mean)
            
        # set the concentration variances
        self.c_vars = self.parameters['c_var_vector']


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibrarySparseBase, self).repr_params
        c_distribution = self.parameters['c_distribution']
        if c_distribution == 'exponential':
            params.append('c=expon(<mean>=%g)' % self.c_means.mean())
        elif c_distribution == 'log-normal':
            params.append('c=lognorm(<mean>=%g, <var>=%g)'
                          % (self.c_means.mean(), self.c_vars.mean()))
        else:
            raise ValueError('Unknown concentration distribution `%s`.'
                             % c_distribution)
        return params


    @classmethod
    def get_random_arguments(cls, **kwargs):
        """ create random args for creating test instances """
        args = super(LibrarySparseBase, cls).get_random_arguments(**kwargs)
        
        # choose concentration distribution
        if 'c_distribution' in kwargs:
            c_distribution = kwargs['c_distribution']
        else:
            c_distribution = np.random.choice(['exponential', 'log-normal'])

        # choose concentration mean
        if kwargs.get('homogeneous_mixture', False):
            c_means = np.full(args['num_substrates'], np.random.random() + 0.5)
        else:
            c_means = np.random.random(args['num_substrates']) + 0.5
            
        # choose concentration variance
        if c_distribution == 'log-normal':
            if kwargs.get('homogeneous_mixture', False):
                c_vars = np.full(args['num_substrates'],
                                 np.random.random() + 0.5)
            else:
                c_vars = np.random.random(args['num_substrates']) + 0.5
        else:
            c_vars = None
            
        args['parameters'].update({'c_distribution': c_distribution,
                                   'c_mean_vector': c_means,
                                   'c_var_vector': c_vars})
        return args
    
    
    @property
    def c_means(self):
        """ return the c_means vector """
        return self._ds
    
    @c_means.setter
    def c_means(self, ds):
        """ sets the substrate c_means """
        if ds is None:
            # initialize with default values, but don't save the parameters
            self._ds = np.ones(self.Ns)
            
        else:
            if any(np.atleast_1d(ds) < 0):
                raise ValueError('Concentration vector must not contain '
                                 'negative entries.')
                
            if np.isscalar(ds):
                self._ds = np.full(self.Ns, ds, np.double)
            elif len(ds) == self.Ns:
                self._ds = np.asarray(ds)
            else:
                raise ValueError('Length of the concentration vector must '
                                 'match the number of substrates.')
            
            # save the values, since they were set explicitly 
            self.parameters['c_mean_vector'] = self._ds
    
    
    @property
    def c_vars(self):
        """ return the c_vars vector """
        if self.parameters['c_distribution'] == 'exponential':
            return self.c_means**2
        else:
            return self._c_vars
    
    @c_vars.setter
    def c_vars(self, variances):
        """ set the c_vars vector """
        if variances is None:
            self._c_vars = np.zeros(self.Ns)
            
        elif self.parameters['c_distribution'] == 'exponential':
            raise RuntimeError('Exponential distributions do not support a '
                               'variance.')
            
        elif self.parameters['c_distribution'] == 'bernoulli':
            raise RuntimeError('Bernoulli distributions do not support a '
                               'variance.')
            
        else:
            if np.isscalar(variances):
                self._c_vars = np.full(self.Ns, variances, np.double)
            else:
                self._c_vars = variances
            # save the values, since they were set explicitly 
            self.parameters['c_var_vector'] = self._c_vars
    
    
    @property
    def concentration_means(self):
        """ return the mean concentration at which each substrate is expected
        on average """
        return self.substrate_probabilities * self.c_means

    
    def get_concentration_distribution(self, i):
        """ returns the concentration distribution for component i """
        c_distribution = self.parameters['c_distribution']
        if c_distribution == 'exponential':
            return stats.expon(scale=self.c_means[i])
        elif c_distribution == 'log-normal':
            return lognorm_mean_var(self.c_means[i], self.c_vars[i])
        elif c_distribution == 'bernoulli':
            return DeterministicDistribution(self.c_means[i])
        else:
            raise ValueError('Unknown concentration distribution `%s`'
                             % c_distribution)

    
    def concentration_statistics(self):
        """ returns statistics for each individual substrate """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        pi = self.substrate_probabilities
        c_means = self.c_means
        ci_mean = pi * c_means
        ci_var = pi * ((1 - pi)*c_means**2 + self.c_vars)
        
        # return the results in a dictionary to be able to extend it later
        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': np.diag(ci_var), 'cov_is_diagonal': True}

    
    @property
    def is_homogeneous_mixture(self):
        """ returns True if the mixture is homogeneous """
        return all(np.allclose(arr, arr.mean())
                   for arr in (self.commonness, self.c_means, self.c_vars))
            
    
    def choose_concentrations(self, scheme, mean_concentration, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `mean_concentration` sets the mean concentration of the individual
            ligands
        """
        
        if scheme == 'const':
            # all substrates are equally likely
            c_means = np.full(self.Ns, mean_concentration, np.double)
                
        elif scheme == 'random_uniform':
            # draw the mean probabilities from a uniform distribution
            c_means = np.random.uniform(0, 2*mean_concentration, self.Ns)
            
        else:
            raise ValueError('Unknown concentration scheme `%s`' % scheme)

        # make sure that the mean concentration is correct
        c_means *= mean_concentration / c_means.mean()
        
        # set the concentration
        self.c_means = c_means
                
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'mean_concentration': mean_concentration}
        c_params.update(kwargs)
        self.parameters['c_mean_parameters'] = c_params

    
        