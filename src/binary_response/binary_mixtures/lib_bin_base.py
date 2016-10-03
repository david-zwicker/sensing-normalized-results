'''
Created on Apr 1, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import logging
import warnings

import numpy as np
from scipy import linalg

from ..library_base import LibraryBase



class LibraryBinaryBase(LibraryBase):
    """ represents a single receptor library. This is a base class that provides
    general functionality and parameter management.
    
    For instance, the class provides a framework for calculating ensemble
    averages, where each time new commonness vectors are chosen randomly
    according to the parameters of the last call to `choose_commonness`.  
    """

    # default parameters that are used to initialize a class if not overwritten
    parameters_default = {
        'commonness_vector': None,      #< chosen substrate commonness
        'commonness_parameters': None,  #< parameters for substrate commonness
        'correlation_matrix': None,     #< chosen substrate correlations
        'correlation_parameters': None, #< parameters for substrate correlations
    }


    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize a receptor library by setting the number of receptors,
        the number of substrates it can respond to, and optional additional
        parameters in the parameter dictionary """
        super(LibraryBinaryBase, self).__init__(num_substrates, num_receptors,
                                                parameters)

        # determine how to initialize the variables
        init_state = self.parameters['initialize_state']
        
        # determine how to initialize the commonness
        init_commonness = init_state.get('commonness', init_state['default'])
        if init_commonness  == 'auto':
            if self.parameters['commonness_parameters'] is None:
                init_commonness = 'exact'
            else:
                init_commonness = 'ensemble'

        # initialize the commonness with the chosen method            
        if init_commonness is None:
            self.commonness = None
            
        elif init_commonness  == 'exact':
            logging.debug('Initialize with given commonness')
            self.commonness = self.parameters['commonness_vector']
            
        elif init_commonness == 'ensemble':
            commonness_parameters = self.parameters['commonness_parameters']
            if commonness_parameters:
                logging.debug('Choose commonness from given parameters')
                self.choose_commonness(**commonness_parameters)
            else:
                logging.warning('Requested to set commonness from parameters, '
                                'but parameters were not supplied.')
                self.commonness = None
                        
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_commonness)
        
        # determine how to initialize the correlations
        init_correlations = init_state.get('correlations',
                                           init_state['default'])
        if init_correlations  == 'auto':
            if self.parameters['correlation_parameters'] is None:
                init_correlations = 'exact'
            else:
                init_correlations = 'ensemble'
                
        # initialize the correlations with the chosen method
        if init_correlations is None:
            self.correlations = None
            
        elif init_correlations == 'exact':
            logging.debug('Initialize with given correlation')
            self.correlations = self.parameters['correlation_matrix']
            
        elif init_correlations == 'ensemble':
            correlation_parameters = self.parameters['correlation_parameters']
            if correlation_parameters:
                logging.debug('Choose correlation from given parameters')
                self.choose_correlations(**correlation_parameters)
            else:
                logging.warning('Requested to set correlation from parameters, '
                                'but parameters were not supplied.')
                self.correlations = None
                
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_correlations)


    @property
    def repr_params(self):
        """ return the important parameters that are shown in __repr__ """
        params = super(LibraryBinaryBase, self).repr_params
        try:
            params.append('m=%g' % self.mixture_size_statistics()['mean'])
        except NotImplementedError:
            # the statistics cannot always be calculated
            params.append('m=?')

        return params


    @classmethod
    def get_random_arguments(cls, homogeneous_mixture=False,
                             mixture_correlated=False, **kwargs):
        """ create random arguments for creating test instances """
        args = super(LibraryBinaryBase, cls).get_random_arguments(**kwargs)
        Ns = args['num_substrates']
        
        if homogeneous_mixture:
            hs = np.full(Ns, np.random.random(), np.double)
        else:
            hs = np.random.random(Ns)
            
        if mixture_correlated:
            Jij = np.random.normal(size=(Ns, Ns))
            np.fill_diagonal(Jij, 0)
            # the matrix will be symmetrize when it is set on the instance 
        else:
            Jij = np.zeros((Ns, Ns))
            
        args['parameters'].update({'commonness_vector': hs,
                                   'correlation_matrix': Jij})
        return args


    @property
    def commonness(self):
        """ return the commonness vector """
        return self._hs
    
    @commonness.setter
    def commonness(self, hs):
        """ sets the commonness and the associated substrate probability """
        if hs is None:
            # initialize with default values, but don't save the parameters
            self._hs = np.zeros(self.Ns)
            self._ps = np.full(self.Ns, 0.5, np.double)
            
        else:
            if len(hs) != self.Ns:
                raise ValueError('Length of the commonness vector must match the '
                                 'number of substrates.')
            self._hs = np.asarray(hs)
            self._ps = 1/(1 + np.exp(-self._hs))
            
            # save the values, since they were set explicitly 
            self.parameters['commonness_vector'] = self._hs
    
    
    @property
    def substrate_probabilities(self):
        """ return the probability of finding each substrate """
        return self._ps
    
    @substrate_probabilities.setter
    def substrate_probabilities(self, ps):
        """ sets the substrate probability and the associated commonness """
        if len(ps) != self.Ns:
            raise ValueError('Length of the probability vector must match the '
                             'number of substrates.')
        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('All probabilities must be within [0, 1]')
        
        with np.errstate(all='ignore'):
            self._hs = np.log(ps) - np.log1p(-ps)
        self._ps = ps
        
        # save the values, since they were set explicitly 
        self.parameters['commonness_vector'] = self._hs
            
    
    @property
    def is_homogeneous_mixture(self):
        """ returns True if the mixture is homogeneous """
        h_i = self.commonness
        return np.allclose(h_i, h_i[0])
            

    @property
    def correlations(self):
        """ return the correlation matrix """
        return self._Jij
    
    @correlations.setter
    def correlations(self, Jij):
        """ sets the correlations """
        if Jij is None:
            # initialize with default values, but don't save the parameters
            self._Jij = np.zeros((self.Ns, self.Ns))
            
        else:
            if Jij.shape != (self.Ns, self.Ns):
                raise ValueError('Dimension of the correlation matrix must be '
                                 'Ns x Ns, where Ns is the number of '
                                 'substrates.')
            self._Jij = np.asarray(Jij)
            
            # symmetrize the matrix by only using the lower triangle
            lower_triangle = np.tril(Jij, -1)
            self._Jij = lower_triangle + lower_triangle.T
            # the diagonal will also be zero now
        
            # save the values, since they were set explicitly 
            self.parameters['correlation_matrix'] = self._Jij
    
    
    @property 
    def is_correlated_mixture(self):
        """ returns True if the mixture has correlations """
        return np.count_nonzero(self.correlations) > 0
    
                
    def mixture_size_distribution(self):
        """ calculates the probabilities of finding a mixture with a given
        number of components. Returns an array of length Ns + 1 of probabilities
        for finding mixtures with the number of components given by the index
        into the array """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')
        
        res = np.zeros(self.Ns + 1)
        res[0] = 1
        # iterate over each substrate and consider its individual probability
        for k, p in enumerate(self.substrate_probabilities, 1):
            res[k] = res[k-1]*p
            res[1:k] = (1 - p)*res[1:k] + res[:k-1]*p
            res[0] = (1 - p)*res[0]
            
        return res


    def mixture_size_statistics(self):
        """ calculates the mean and the standard deviation of the number of
        components in mixtures """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        # calculate basic statistics
        prob_s = self.substrate_probabilities
        m_mean = np.sum(prob_s)
        m_var = np.sum(prob_s/(1 + np.exp(self._hs)))
        
        # return the results in a dictionary to be able to extend it later
        return {'mean': m_mean, 'std': np.sqrt(m_var), 'var': m_var}


    def mixture_entropy(self):
        """ return the entropy in the mixture distribution """
        if self.is_correlated_mixture:
            raise NotImplementedError('Not implemented for correlated mixtures')

        pi = self.substrate_probabilities
        return -np.sum(pi*np.log2(pi) + (1 - pi)*np.log2(1 - pi))

    
    def choose_commonness(self, scheme, mean_mixture_size, **kwargs):
        """ picks a commonness vector according to the supplied parameters:
        `mean_mixture_size` determines the mean number of components in each
        mixture. The commonness vector is then chosen according to the given  
        `scheme`, which can be any of the following:
            `const`: all substrates have equal probability
            `single`: the first substrate has a different probability, which can
                either be specified directly by supplying the parameter `p1` or
                the `ratio` between p1 and the probabilities of the other
                substrates can be specified.
            `geometric`: the probability of substrates decreases by a factor of
                `alpha` from each substrate to the next. The ratio `alpha`
                between subsequent probabilities should be supplied as a
                keyword parameter.
            `linear`: the probability of substrates decreases linearly.
            `random_uniform`: the probability of substrates is chosen from a
                uniform distribution with given mean and maximal variance.
                An optional parameter `sigma` with values between 0 and 1 can be
                given in order to restrict the uniform distribution to a
                fraction of its maximal width. 
        """
        if not 0 <= mean_mixture_size <= self.Ns:
            raise ValueError('The mean mixture size must be between 0 and the '
                             'number of ligands/substrates (%d).' % self.Ns) 
        
        if scheme == 'const':
            # all substrates are equally likely
            ps = np.full(self.Ns, mean_mixture_size/self.Ns, np.double)
        
        elif scheme == 'single':
            # the first substrate has a different commonness than the others
            ps = np.empty(self.Ns)
            if 'p1' in kwargs:
                # use the given probability for the first substrate
                ps[0] = kwargs['p1']
                ps[1:] = (mean_mixture_size - ps[0]) / (self.Ns - 1)
                 
            elif 'p_ratio' in kwargs:
                # use the given ratio between the first and the other substrates
                ratio = kwargs['p_ratio']
                denom = self.Ns + ratio - 1
                ps[0] = mean_mixture_size * ratio / denom
                ps[1:] = mean_mixture_size / denom
                
            else:
                raise ValueError('Either `p1` or `p_ratio` must be given')
            
        elif scheme == 'geometric':
            # substrates have geometrically decreasing commonness
            try: 
                alpha = kwargs['alpha']
            except KeyError:
                raise ValueError(
                    'The ratio `alpha` between subsequent probabilities must '
                    ' be supplied as a keyword parameter'
                )

            if alpha == 1:
                p0 = mean_mixture_size/self.Ns
            else:
                p0 = mean_mixture_size * (1 - alpha) / (1 - alpha**self.Ns)
                
            if p0 > 1:
                raise ValueError('It is not possible to choose commonness '
                                 'parameters such that the mean mixture size '
                                 'of %d components is %g for alpha=%g'
                                 % (self.Ns, mean_mixture_size, alpha))
                
            ps = p0 * alpha**np.arange(self.Ns)
            
        elif scheme == 'linear':
            # substrates have a linear decreasing probability
            if mean_mixture_size <= 0.5*self.Ns:
                a, b = 0, 2*mean_mixture_size/self.Ns
            else:
                a, b = 2*mean_mixture_size/self.Ns - 1, 1
                
            ps = np.linspace(a, b, self.Ns)
            
        elif scheme == 'random_uniform':
            # substrates have probability chosen from a uniform distribution

            # determine the bounds for the maximal random distribution
            if mean_mixture_size <= 0.5*self.Ns:
                a, b = 0, 2*mean_mixture_size/self.Ns
            else:
                a, b = 2*mean_mixture_size/self.Ns - 1, 1

            # determine the width of the distribution 
            sigma = kwargs.pop('sigma', 1)
            if 0 <= sigma <= 1: 
                mean = 0.5*(a + b)
                halfwidth = 0.5*sigma*(b - a)
                a = mean - halfwidth
                b = mean + halfwidth
            else:
                raise ValueError("The width sigma of the uniform distribution "
                                 "must be chosen from the interval [0, 1].")
                
            # choose random probabilities
            ps = np.random.uniform(a, b, size=self.Ns)
            ps_mean = ps.sum()
            
            # correct the probabilities to ensure the mean
            if ps_mean < mean_mixture_size:
                # increase ps to match mean 
                ps_c = 1 - ps #< consider the compliment
                ps_c *= (self.Ns - mean_mixture_size)/(self.Ns - ps_mean)
                ps = 1 - ps_c
            else:
                # decrease ps to match mean
                ps *= mean_mixture_size/ps_mean
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)
        
        # set the probability which also calculates the commonness and saves
        # the values in the parameters dictionary
        self.substrate_probabilities = ps
        
        # we additionally store the parameters that were used for this function
        c_params = {'scheme': scheme, 'mean_mixture_size': mean_mixture_size}
        c_params.update(kwargs)
        self.parameters['commonness_parameters'] = c_params  


    def choose_correlations(self, scheme, magnitude, **kwargs):
        """ picks a correlation matrix according to the supplied parameters:
        `magnitude` determines the magnitude of the correlations, which are
        drawn from the random distribution indicated by `scheme`: 
            `const`: all correlations are equally to magnitude
            `block_uniform`: block diagonal form with uniform block size
                distribution determined by the parameter `count`, which sets
                the number of blocks
            `random_binary`: the correlations are drawn from a binary
                distribution which is either 1 or -1 times magnitude
            `random_uniform`: the correlations are drawn from a uniform
                distribution from [-magnitude, magnitude]
            `random_normal`: the correlations are drawn from a normal
                distribution with standard deviation `magnitude`
            `random_sparse` sparse correlations matrix, where the density is
                set by the parameter `density`
        """
        shape = (self.Ns, self.Ns)
        
        if 'diagonal_zero' in kwargs:
            warnings.warn("diagonal_zero is deprecated since it is now implied "
                          "by default.", DeprecationWarning)
            kwargs.pop('diagonal_zero')

        corr_params = {'scheme': scheme, 'magnitude': magnitude}

        if scheme == 'const':
            # all correlations are equal
            Jij = np.full(shape, magnitude, np.double)
            
        elif scheme == 'blocks_uniform':
            # choose block diagonal form with uniform block sizes
            count = kwargs.pop('count', self.Ns)
            corr_params['count'] = count
            if count >= self.Ns:
                Jij = np.eye(self.Ns) * magnitude
            else:
                # create `count` blocks of almost even size
                sizes = np.full(count, self.Ns // count, np.int)
                sizes[:self.Ns % count] += 1
                blocks = [magnitude * np.ones((size, size)) for size in sizes]
                Jij = linalg.block_diag(*blocks)
        
        elif scheme == 'random_binary':
            # all correlations are binary times magnitude
            Jij = np.random.choice((-1, 1), shape) * magnitude
            
        elif scheme == 'random_uniform':
            # all correlations are uniformly distributed
            Jij = np.random.uniform(-magnitude, magnitude, shape)
            
        elif scheme == 'random_normal':
            # all correlations are uniformly distributed
            Jij = np.random.normal(scale=magnitude, size=shape)
        
        elif scheme == 'random_sparse':
            density = kwargs.pop('density', 0.5)
            corr_params['density'] = density
            Jij = magnitude * (np.random.random(shape) < density)
            
        else:
            raise ValueError('Unknown commonness scheme `%s`' % scheme)

        # set the correlations, which will only use the lower triangle of Jij
        # and thus symmetrize the matrix and make sure that the diagonal is zero
        self.correlations = Jij
        
        # we additionally store the parameters that were used for this function
        self.parameters['correlation_parameters'] = corr_params  
        
        # raise an error if keyword arguments have not been used
        if len(kwargs) > 0:
            raise ValueError('The following keyword arguments have not been '
                             'used: %s' % str(kwargs))         

