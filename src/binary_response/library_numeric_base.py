'''
Created on Sep 10, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import functools
import logging
import time

import numpy as np
from scipy import optimize

from utils.math.distributions import lognorm_mean, loguniform_mean
from utils.math import is_pos_semidef



class LibraryNumericMixin(object):
    """ Mixin class that defines functions that are useful for numerical 
    calculations.
    
    The iteration over mixtures must be implemented in the subclass. Here, we 
    expect the methods `_sample_mixtures` and `_sample_steps` to exist, which
    are a generator of mixtures and its expected length, respectively.    
    """
    
    def __init__(self, num_substrates, num_receptors, parameters=None):
        """ initialize the sensitivity matrix """
        # the call to the inherited method also sets the default parameters from
        # this class
        super(LibraryNumericMixin, self).__init__(num_substrates, num_receptors,
                                                  parameters)        

        # determine how to initialize the variables
        init_state = self.parameters['initialize_state']
        
        # determine how to initialize the commonness
        init_sensitivity = init_state.get('sensitivity', init_state['default'])
        if init_sensitivity  == 'auto':
            if self.parameters['sensitivity_matrix'] is not None:
                init_sensitivity = 'exact'
            elif self.parameters['sensitivity_matrix_params'] is not None:
                init_sensitivity = 'ensemble'
            else:
                init_sensitivity = 'zero'

        # initialize the commonness with the chosen method            
        if init_sensitivity is None or init_sensitivity == 'zero':
            self.sens_mat = np.zeros((self.Nr, self.Ns), np.uint8)

        elif init_sensitivity  == 'exact':
            logging.debug('Initialize with given sensitivity matrix')
            sens_mat = self.parameters['sensitivity_matrix']
            if sens_mat is None:
                logging.warning('Sensitivity matrix was not given. Initialize '
                                'empty matrix.')
                self.sens_mat = np.zeros((self.Nr, self.Ns), np.uint8)
            else:
                self.sens_mat = sens_mat.copy()
            
        elif init_sensitivity == 'ensemble':
            logging.debug('Choose sensitivity matrix from given parameters')
            sens_params = self.parameters['sensitivity_matrix_params']
            if sens_params is None:
                logging.warning('Parameters for sensitivity matrix were not '
                                'specified. Initialize empty matrix.')
                self.sens_mat = np.zeros((self.Nr, self.Ns), np.uint8)
            else:
                self.choose_sensitivity_matrix(**sens_params)
                    
        else:
            raise ValueError('Unknown initialization protocol `%s`' % 
                             init_sensitivity)

        assert self.sens_mat.shape == (self.Nr, self.Ns)
    
    
    
    def concentration_statistics(self, method='auto', **kwargs):
        """ calculates mixture statistics using a metropolis algorithm
        Returns the mean concentration, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.concentration_statistics_monte_carlo(**kwargs)
        elif method == 'estimate':
            return self.concentration_statistics_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
            
            
    def concentration_statistics_monte_carlo(self):
        """ calculates mixture statistics using a metropolis algorithm """
        count = 0
        hist1d = np.zeros(self.Ns)
        hist2d = np.zeros((self.Ns, self.Ns))

        # sample mixtures uniformly
        # FIXME: use better online algorithm that is less prone to canceling
        for c in self._sample_mixtures():
            count += 1
            hist1d += c
            hist2d += np.outer(c, c)
        
        # calculate the frequency and the correlations 
        ci_mean = hist1d/count
        cij_corr = hist2d/count - np.outer(ci_mean, ci_mean)
        
        ci_var = np.diag(cij_corr)
        return {'mean': ci_mean, 'std': np.sqrt(ci_var), 'var': ci_var,
                'cov': cij_corr}

        
    def excitation_statistics(self, method='auto', ret_correlations=True,
                              **kwargs):
        """ calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.

        `method` can be one of [monte_carlo', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.excitation_statistics_monte_carlo(ret_correlations,
                                                          **kwargs)
        elif method == 'estimate':
            return self.excitation_statistics_estimate(**kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
        
        
    def excitation_statistics_monte_carlo(self, ret_correlations=False):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        
        The algorithms used here have been taken from
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat

        if ret_correlations:
            # calculate the mean and the covariance matrix
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            enm_cov = np.zeros((self.Nr, self.Nr))
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = (e_n - en_mean) / count
                en_mean += delta
                enm_cov += ((count - 1) * np.outer(delta, delta)
                            - enm_cov / count)
                
            # calculate the requested statistics
            if count < 2:
                enm_cov.fill(np.nan)
            else:
                enm_cov *= count / (count - 1)
            
            en_var = np.diag(enm_cov)
            
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                    'cov': enm_cov}
            
        else:
            # only calculate the mean and the variance
    
            # prepare variables holding the necessary data
            en_mean = np.zeros(self.Nr)
            en_square = np.zeros(self.Nr)
            
            # sample mixtures and safe the requested data
            for count, c_i in enumerate(self._sample_mixtures(), 1):
                e_n = np.dot(S_ni, c_i)
                delta = e_n - en_mean
                en_mean += delta / count
                en_square += delta * (e_n - en_mean)
                
            # calculate the requested statistics
            if count < 2:
                en_var.fill(np.nan)
            else:
                en_var = en_square / (count - 1)
    
            return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var}
                            
    
    def excitation_statistics_estimate(self, **kwargs):
        """
        calculates the statistics of the excitation of the receptors.
        Returns the mean excitation, the variance, and the covariance matrix.
        """
        c_stats = self.concentration_statistics_estimate(**kwargs)
        
        # calculate statistics of e_n = \sum_i S_ni * c_i        
        S_ni = self.sens_mat
        en_mean = np.dot(S_ni, c_stats['mean'])
        cov_is_diagonal = c_stats.get('cov_is_diagonal', False)
        if cov_is_diagonal:
            enm_cov = np.einsum('ni,mi,i->nm', S_ni, S_ni, c_stats['var'])
        else:
            enm_cov = np.einsum('ni,mj,ij->nm', S_ni, S_ni, c_stats['cov'])
        en_var = np.diag(enm_cov)
        
        return {'mean': en_mean, 'std': np.sqrt(en_var), 'var': en_var,
                'cov': enm_cov}
        
        
    def receptor_activity(self, method='auto', ret_correlations=False, **kwargs):
        """ calculates the average activity of each receptor
        
        `method` can be one of [monte_carlo', 'estimate'].
        `ret_correlations` determines whether the correlations are returned
            alongside the mean activities. Note that the correlations are not
            the connected correlation coefficients but the bare expectation
            values <a_n a_m>.
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.receptor_activity_monte_carlo(ret_correlations, **kwargs)
        elif method == 'estimate':
            return self.receptor_activity_estimate(ret_correlations, **kwargs)
        else:
            raise ValueError('Unknown method `%s`.' % method)
     
            
    def receptor_activity_monte_carlo(self, ret_correlations=False):
        """ calculates the average activity of each receptor
        
        `ret_correlations` determines whether the correlations are returned
            alongside the mean activities. Note that the correlations are not
            the connected correlation coefficients but the bare expectation
            values <a_n a_m>.        
        """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        S_ni = self.sens_mat

        r_n = np.zeros(self.Nr)
        if ret_correlations:
            r_nm = np.zeros((self.Nr, self.Nr))
        
        for c_i in self._sample_mixtures():
            e_n = np.dot(S_ni, c_i)
            a_n = (e_n >= 1)
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
        """ estimates the average activity of each receptor

        `ret_correlations` determines whether the correlations are returned
            alongside the mean activities. Note that the correlations are not
            the connected correlation coefficients but the bare expectation
            values <a_n a_m>.
        `excitation_model` defines what model is used to estimate the excitation
            statistics.
        """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor activity
        r_n = self._estimate_qn_from_en(en_stats,
                                        excitation_model=excitation_model)
        if clip:
            np.clip(r_n, 0, 1, r_n)

        if ret_correlations:
            # calculate the correlated activity 
            q_nm = self._estimate_qnm_from_en(en_stats)
            r_nm = np.outer(r_n, r_n) + q_nm
            if clip:
                np.clip(r_nm, 0, 1, r_nm)

            return r_n, r_nm
        else:
            return r_n   
        

    def receptor_pearson_correlation(self, method='auto', ret=None,
                                     invalid_value=np.nan):
        """ calculates Pearson's correlation coefficient between receptors.
    
        `method` determines the method used to calculated the receptor activity
        `ret` is a list which determines which values are returned. If ret is
            None, the mean and the standard deviation of the correlation
            coefficients are returned. If ret is `all`, all possible values are
            returned.
        `invalid_value` defines the value that is returned for invalid values
            of the correlation coefficient. The default is `numpy.nan`
            
        Note that `pearson_mean` and `pearson_std` are only calculated for valid
        entries of the Pearson's correlation coefficient matrix. 
        """
        if ret is None:
            ret = ['pearson_mean', 'pearson_std']
        elif ret == 'all':
            ret = ['activity', 'covariance', 'pearson', 'pearson_mean',
                   'pearson_std']
        ret = set(ret)

        # calculate the statistics of the receptor activities        
        r_n, r_nm = self.receptor_activity(method, ret_correlations=True)
        # calculate the covariance matrix
        r_cov = r_nm - np.outer(r_n, r_n)
        
        # calculate the standard deviation of the activities
        r_std = np.sqrt(np.diag(r_cov))
        
        # calculate Pearson's correlation coefficient for non-zero entries
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = r_cov / np.outer(r_std, r_std)
        
        # get all entries of the upper triangle
        corr_tri = corr[np.triu_indices_from(corr, 1)]
        is_valid = np.any(np.isfinite(corr_tri))

        result = {}
        if 'activity' in ret:
            result['activity'] = r_n
            ret.remove('activity')
            
        if 'covariance' in ret:
            result['covariance'] = r_cov
            ret.remove('covariance')
            
        if 'pearson' in ret:
            # set invalid values if requested
            if not np.isnan(invalid_value):
                idx_invalid = (r_std == 0)
                corr[idx_invalid, :] = corr[:, idx_invalid] = invalid_value
                np.fill_diagonal(corr, 1)
            
            result['pearson'] = corr
            ret.remove('pearson')
            
        if 'pearson_mean' in ret:
            if is_valid:
                result['pearson_mean'] = np.nanmean(corr_tri)
            else:
                result['pearson_mean'] = invalid_value
            ret.remove('pearson_mean')
            
        if 'pearson_std' in ret:
            if is_valid:
                result['pearson_std'] = np.nanstd(corr_tri)
            else:
                result['pearson_std'] = invalid_value
            ret.remove('pearson_std')
            
        if ret:
            raise ValueError('Do not know how to calculate %s' % ret)
            
        return result


    def receptor_crosstalk(self, method='auto', ret_receptor_activity=False,
                           clip=False, **kwargs):
        """ calculates the crosstalk between receptors, which is quantified by
        the covariance cov(a_n, a_m) = <a_n a_m> - <a_n><a_m> between the 
        receptors.
        
        `method` can be ['brute_force', 'monte_carlo', 'estimate', 'auto'].
            If it is 'auto' than the method is chosen automatically based on the
            problem size.
        `ret_receptor_activity` determines whether the mean receptor activities
            are also returned. This will return a tuple (mean_a, cov_a).
        """
        if method == 'estimate':
            kwargs['clip'] = False

        # calculate receptor activities with the requested `method`            
        r_n, r_nm = self.receptor_activity(method, ret_correlations=True,
                                           **kwargs)
        
        # calculate receptor crosstalk from the observed probabilities
        q_nm = r_nm - np.outer(r_n, r_n)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)
        
        if ret_receptor_activity:
            return r_n, q_nm # q_n = r_n
        else:
            return q_nm

        
    def receptor_crosstalk_estimate(self, ret_receptor_activity=False,
                                    excitation_model='default', clip=False):
        """ estimates the crosstalk between receptors, which is quantified by
        the covariance cov(a_n, a_m) = <a_n a_m> - <a_n><a_m> between the 
        receptors.
        """
        en_stats = self.excitation_statistics_estimate()

        # calculate the receptor crosstalk
        q_nm = self._estimate_qnm_from_en(en_stats)
        if clip:
            np.clip(q_nm, 0, 1, q_nm)

        if ret_receptor_activity:
            # calculate the receptor activity
            q_n = self._estimate_qn_from_en(en_stats, excitation_model)
            if clip:
                np.clip(q_n, 0, 1, q_n)

            return q_n, q_nm
        else:
            return q_nm        
        
        
    def mutual_information(self, method='auto', ret_prob_activity=False,
                           **kwargs):
        """ calculate the mutual information of the receptor array.

        `method` can be one of [monte_carlo', 'moments', 'estimate'].
        """
        if method == 'auto':
            method = 'monte_carlo'
                
        if method == 'monte_carlo' or method == 'monte-carlo':
            return self.mutual_information_monte_carlo(ret_prob_activity)
        elif method == 'moments':
            return self.mutual_information_from_moments(ret_prob_activity)
        elif method == 'estimate':
            return self.mutual_information_estimate(ret_prob_activity, **kwargs)
        else:
            raise ValueError('Unknown method for determining mutual '
                             'information `%s`.' % method)

                                                   
    def mutual_information_monte_carlo(self, ret_prob_activity=False):
        """ calculate the mutual information using a monte carlo strategy. The
        number of steps is given by the model parameter 'monte_carlo_steps' """
        # prevent integer overflow in collecting activity patterns
        assert self.Nr <= self.parameters['max_num_receptors'] <= 63

        base = 2 ** np.arange(0, self.Nr)

        # sample mixtures according to the probabilities of finding
        # substrates
        count_a = np.zeros(2**self.Nr)
        for c in self._sample_mixtures():
            # get the activity vector ...
            a = (np.dot(self.sens_mat, c) >= 1)
            # ... and represent it as a single integer
            a_id = np.dot(base, a)
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


    def mutual_information_from_moments(self, ret_prob_activity=False):
        """ calculate the mutual information using the first and second moments
        of the receptor activity distribution """
        # determine the moments of the receptor activity distribution
        r_n, r_nm = self.receptor_activity(ret_correlations=True)
        
        # determine MI from these moments
        MI = self._estimate_MI_from_r_values(r_n, r_nm)
        
        if ret_prob_activity:
            return MI, r_n
        else:
            return MI

                    
    def mutual_information_estimate(self, ret_prob_activity=False,
                                    excitation_model='default',
                                    mutual_information_method='default',
                                    clip=True):
        """ returns a simple estimate of the mutual information.
        `clip` determines whether the approximated probabilities should be
            clipped to [0, 1] before being used to calculate the mutual info.
        """
        q_n, q_nm = self.receptor_crosstalk_estimate(
            ret_receptor_activity=True, \
            excitation_model=excitation_model,
            clip=clip
        )
        
        # calculate the approximate mutual information
        MI = self._estimate_MI_from_q_values(
                                    q_n, q_nm, method=mutual_information_method)
        
        if clip:
            MI = np.clip(MI, 0, self.Nr)
        
        if ret_prob_activity:
            return MI, q_n
        else:
            return MI
        

            
def get_sensitivity_matrix(Nr, Ns, distribution, mean_sensitivity=1,
                           receptor_factors=None,  ensure_mean=False,
                           ret_params=True, **kwargs):
    """ creates a sensitivity matrix with the given properties
        `Nr` is the number of receptors
        `Ns` is the number of substrates/ligands
        `distribution` determines the distribution from which we choose the
            entries of the sensitivity matrix
        `mean_sensitivity` should in principle set the mean sensitivity,
            although there are some exceptional distributions. For instance,
            for binary distributions `mean_sensitivity` sets the
            magnitude of the entries that are non-zero.
        `receptor_factors` is an optional array of pre-factors that are applied
            for each row of the sensitivity matrix after the random numbers have
            been drawn from the distribution, but before `ensure_mean` is
            applied.
        `ensure_mean` makes sure that the mean of the matrix is indeed equal to
            `mean_sensitivity`
        `ret_params` determines whether a dictionary with the parameters that
            lead to the calculated sensitivity is also returned
        Some distributions might accept additional parameters, which can be
        supplied in the dictionary `parameters`.
    """
    if mean_sensitivity <= 0:
        raise ValueError('mean_sensitivity must be positive.')
    
    shape = (Nr, Ns)

    sens_mat_params = {'distribution': distribution,
                       'mean_sensitivity': mean_sensitivity,
                       'receptor_factors': receptor_factors,
                       'ensure_mean': ensure_mean}

    if distribution == 'const':
        # simple constant matrix
        sens_mat = np.full(shape, mean_sensitivity, np.double)

    elif distribution == 'binary':
        # choose a binary matrix with a typical scale
        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            S_mean2 = mean_sensitivity ** 2
            density = S_mean2 / (S_mean2 + standard_deviation**2)
        elif 'density' in kwargs:
            density = kwargs.pop('density')
            standard_deviation = mean_sensitivity * np.sqrt(1/density - 1)
        else:
            standard_deviation = 1
            S_mean2 = mean_sensitivity ** 2
            density = S_mean2 / (S_mean2 + standard_deviation**2)

        if density > 1:
            raise ValueError('Standard deviation is too large.')
            
        sens_mat_params['standard_deviation'] = standard_deviation
        
        if density == 0:
            # simple case of empty matrix
            sens_mat = np.zeros(shape)
            
        elif density >= 1:
            # simple case of full matrix
            sens_mat = np.full(shape, mean_sensitivity, np.double)
            
        else:
            # choose receptor substrate interaction randomly and don't worry
            # about correlations
            S_scale = mean_sensitivity / density
            nonzeros = (np.random.random(shape) < density)
            sens_mat = S_scale * nonzeros 

    elif distribution == 'log_normal' or distribution == 'log-normal':
        # log normal distribution
        if 'spread' in kwargs:
            logging.warning('Deprecated argument `spread`. Use `width` instead')
            kwargs.setdefault('width', kwargs['spread'])

        if 'variance' in kwargs:
            kwargs['standard_deviation'] = np.sqrt(kwargs.pop('variance'))
        
        if 'standard_deviation' in kwargs:
            standard_deviation = kwargs.pop('standard_deviation')
            cv = standard_deviation / mean_sensitivity 
            width = np.sqrt(np.log(cv**2 + 1))
            
        elif 'width' in kwargs:
            width = kwargs.pop('width')
            cv = np.sqrt(np.exp(width**2) - 1)
            standard_deviation = mean_sensitivity * cv
            
        else:
            standard_deviation = 1
            cv = standard_deviation / mean_sensitivity
            width = np.sqrt(np.log(cv**2 + 1))

        correlation = kwargs.pop('correlation', 0)
        sens_mat_params['standard_deviation'] = standard_deviation
        sens_mat_params['correlation'] = correlation

        if width == 0 and correlation == 0:
            # edge case without randomness
            sens_mat = np.full(shape, mean_sensitivity, np.double)

        elif correlation != 0:
            # correlated receptors
            mu = np.log(mean_sensitivity) - 0.5 * width**2
            mean = np.full(Nr, mu, np.double)
            cov = np.full((Nr, Nr), correlation * width**2, np.double)
            np.fill_diagonal(cov, width**2)
            vals = np.random.multivariate_normal(mean, cov, size=Ns).T
            sens_mat = np.exp(vals)

        else:
            # uncorrelated receptors
            dist = lognorm_mean(mean_sensitivity, width)
            sens_mat = dist.rvs(shape)
            
    elif distribution == 'log_uniform' or distribution == 'log-uniform':
        # log uniform distribution
        width = kwargs.pop('width', 1)
        sens_mat_params['width'] = width

        if width == 0:
            sens_mat = np.full(shape, mean_sensitivity, np.double)
        else:
            dist = loguniform_mean(mean_sensitivity, np.exp(width))
            sens_mat = dist.rvs(shape)
        
    elif distribution == 'normal':
        # normal distribution
        width = kwargs.pop('width', 1)
        correlation = kwargs.pop('correlation', 0)
        sens_mat_params['width'] = width
        sens_mat_params['correlation'] = correlation

        if width == 0 and correlation == 0:
            # edge case without randomness
            sens_mat = np.full(shape, mean_sensitivity)
            
        elif correlation != 0:
            # correlated receptors
            mean = np.full(Nr, mean_sensitivity, np.double)
            cov = np.full((Nr, Nr), correlation * width**2, np.double)
            np.fill_diagonal(cov, width**2)
            if not is_pos_semidef(cov):
                raise ValueError('The specified correlation leads to a '
                                 'correlation matrix that is not positive '
                                 'semi-definite.')
            vals = np.random.multivariate_normal(mean, cov, size=Ns)
            sens_mat = vals.T

        else:
            # uncorrelated receptors
            sens_mat = np.random.normal(loc=mean_sensitivity, scale=width,
                                        size=shape)

    elif distribution == 'uniform':
        # uniform sensitivity distribution
        S_min = kwargs.pop('S_min', 0)
        S_max = 2 * mean_sensitivity - S_min
        
        # choose random sensitivities
        sens_mat = np.random.uniform(S_min, S_max, size=shape)
        
    elif distribution == 'gamma':
        raise NotImplementedError
        
    else:
        raise ValueError('Unknown distribution `%s`' % distribution)
        
    if receptor_factors is not None:
        # apply weights to the individual receptors 
        sens_mat *= receptor_factors[:, None]
        
    if ensure_mean:
        # make sure that the mean sensitivity is exactly as given
        sens_mat *= mean_sensitivity / sens_mat.mean()

    # raise an error if keyword arguments have not been used
    if len(kwargs) > 0:
        raise ValueError('The following keyword arguments have not been '
                         'used: %s' % str(kwargs)) 
    
    if ret_params:    
        # return the parameters determining this sensitivity matrix
        return sens_mat, sens_mat_params
    else:
        return sens_mat
    

def _optimize_continuous_library_single(model, target, direction='max',
                                        steps=100, method='cma', ret_info=False,
                                        args=None, verbose=False):
    """ optimizes the current library to maximize the result of the target
    function using gradient descent. By default, the function returns the
    best value and the associated sensitivity matrix as result.        
    """
    # get the target function to call
    target_function = getattr(model, target)
    if args is not None:
        target_function = functools.partial(target_function, **args)

    # define the cost function
    if direction == 'min':
        def cost_function(sens_mat_flat):
            """ cost function to minimize """
            model.sens_mat.flat = sens_mat_flat.flat
            return target_function()
        
    elif direction == 'max':
        def cost_function(sens_mat_flat):
            """ cost function to minimize """
            model.sens_mat.flat = sens_mat_flat.flat
            return -target_function()
        
    else:
        raise ValueError('Unknown optimization direction `%s`' % direction)

    if ret_info:
        # store extra information
        start_time = time.time()
        info = {'values': []}
        
        cost_function_inner = cost_function
        def cost_function(sens_mat_flat):
            """ wrapper function to store calculated costs """
            cost = cost_function_inner(sens_mat_flat)
            info['values'].append(cost)
            return cost
    
    if method == 'cma':
        # use Covariance Matrix Adaptation Evolution Strategy algorithm
        try:
            import cma  # @UnresolvedImport
        except ImportError:
            raise ImportError('The module `cma` is not available. Please '
                              'install it using `pip install cma` or '
                              'choose a different optimization method.')
        
        # prepare the arguments for the optimization call    
        x0 = model.sens_mat.flat
        sigma = 0.5 * np.mean(x0) #< initial step size
        options = {'maxfevals': steps,
                   'bounds': [0, np.inf],
                   'verb_disp': 100 * int(verbose),
                   'verb_log': 0}
        
        # call the optimizer
        res = cma.fmin(cost_function, x0, sigma, options=options)
        
        # get the result
        state_best = res[0].reshape((model.Nr, model.Ns))
        value_best = res[1]
        if ret_info: 
            info['states_considered'] = res[3]
            info['iterations'] = res[4]

    else:
        # use the standard scipy function
        res = optimize.minimize(cost_function, model.sens_mat.flat,
                                method=method, options={'maxiter': steps})
        value_best =  res.fun
        state_best = res.x.reshape((model.Nr, model.Ns))
        if ret_info: 
            info['states_considered'] = res.nfev
            info['iterations'] = res.nit
        
    if direction == 'max':
        value_best *= -1
    
    model.sens_mat = state_best.copy()

    if ret_info:
        info['total_time'] = time.time() - start_time    
        info['performance'] = info['states_considered'] / info['total_time']
        return value_best, state_best, info
    else:
        return value_best, state_best   



def _optimize_continuous_library_parallel(model, target, direction='max',
                                          steps=100, method='cma-parallel',
                                          ret_info=False, args=None,
                                          verbose=False):
    """ optimizes the current library to maximize the result of the target
    function using gradient descent. By default, the function returns the
    best value and the associated sensitivity matrix as result.        
    """
    if ret_info:
        # store extra information
        start_time = time.time()
        info = {'values': []}
    
    if method == 'cma-parallel':
        # use playdoh framework to do CMA-ES
        try:
            import playdoh  # @UnresolvedImport
        except ImportError:
            raise ImportError('The module `playdoh` is not available. Please '
                              'install it using `pip install playdoh` or '
                              'choose a different optimization method.')
            
        # determine parameters
        dim = model.Ns * model.Nr
        popsize = 4 + int(3*np.log(dim)) #< default CMA-ES scaling

        class TargetClass(playdoh.Fitness):
            """ class that is used for optimization """
            
            def initialize(self, model_class, init_arguments):
                """ initialize the model in the class """
                self.model = model_class(**init_arguments)
                # prepare the target function
                target_function = getattr(self.model, target)
                if args is None:
                    self.target_function = target_function
                else:
                    self.target_function = functools.partial(target_function,
                                                             **args)
        
            def evaluate(self, sens_mat_flat):
                """ evaluate the fitness of a given class """
                result = np.empty(sens_mat_flat.shape[1])
                for i in range(sens_mat_flat.shape[1]):
                    self.model.sens_mat.flat = sens_mat_flat[:, i]
                    result[i] = self.target_function()
                return result

        # find the right function to use for optimization
        if direction == 'min':
            optimize = playdoh.minimize
        elif direction == 'max':
            optimize = playdoh.maximize
        else:
            raise ValueError('Unknown optimization direction `%s`' % direction)

        Sopt = 1 / model.concentration_means.sum()
        initrange = np.tile(np.array([0, 3 * Sopt], np.double), (dim, 1))
            
        # do the optimization
        results = optimize(TargetClass,
                           popsize=popsize,     # size of the population
                           maxiter=steps,       # maximum number of iterations
                           cpu=playdoh.MAXCPU,  # number of CPUs to use on the local machine
                           algorithm=playdoh.CMAES,
                           args=(model.__class__, model.init_arguments),
                           initrange=initrange)
                    
        value_best = results.best_fit
        state_best = results.best_pos.reshape((model.Nr, model.Ns))
                                              
    else:
        raise ValueError('Unknown optimization method `%s`' % method)
        
    if direction == 'max':
        value_best *= -1
    
    model.sens_mat = state_best.copy()

    if ret_info:
        info['total_time'] = time.time() - start_time    
        info['performance'] = info['states_considered'] / info['total_time']
        return value_best, state_best, info
    else:
        return value_best, state_best
    
    

def optimize_continuous_library(*args, **kwargs):
    """ optimizes the current library to maximize the result of the target
    function using gradient descent. By default, the function returns the
    best value and the associated sensitivity matrix as result.        
    """
    method = kwargs.get('method', 'cma')
    if 'parallel' in method:
        return _optimize_continuous_library_parallel(*args, **kwargs)
    else:
        return _optimize_continuous_library_single(*args, **kwargs)
    
    
    