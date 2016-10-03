#!/usr/bin/env python
'''
Created on Apr 3, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys
import os.path
# append base path to sys.path
script_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(script_path, '..', '..'))

import argparse
import datetime
import itertools
import math
import multiprocessing as mp

import six.moves.cPickle as pickle

from binary_response import LibrarySparseNumeric, LibrarySparseLogNormal


# introduce global variable for keeping track of the number the started jobs
jobs_started = mp.Value('I', 0)



def optimize_library(parameters):
    """ optimize receptors of the system described by `parameters` """
    global jobs_started

    if parameters['progress']:
        jobs_started.value += 1
        job_id, job_count = jobs_started.value, parameters['job_count']
        progress = math.floor(100 * job_id / job_count)
        print('Started job %d of %d (%d%%) at %s' %
              (job_id, job_count, progress, datetime.datetime.now()))
        sys.stdout.flush() #< make output appear immediately
    
    # setup the numerical model that we use for optimization
    model = LibrarySparseNumeric(
        parameters['Ns'], parameters['Nr'],
        parameters={'verbosity': 0 if parameters['quite'] else 1,
                    'c_distribution': parameters['concentration-distribution']}
    )
    model.choose_commonness(parameters['mixture-scheme'],
                            parameters['mixture-size'])
    model.choose_concentrations(parameters['concentration-scheme'],
                                parameters['concentration-mean'])
    model.choose_correlations(parameters['correlation-scheme'],
                              parameters['correlation-magnitude'])
    if model.parameters['c_distribution'] != 'exponential':
        model.c_vars = parameters['concentration-var']
    
    # get optimal log-normal matrix as a starting point
    theory = LibrarySparseLogNormal.from_other(model, width=2)
    library_opt = theory.get_optimal_library(fixed_parameter='width')
    
    # choose the method for calculating the mutual information
    if parameters['MI-method'] == 'numeric':
        target = 'mutual_information_monte_carlo'
        args = {}
    elif parameters['MI-method'] == 'approx':
        target = 'mutual_information_estimate'
        args = {}
    elif parameters['MI-method'] == 'approx-gaussian':
        target = 'mutual_information_estimate'
        args = {'excitation_model': 'gaussian'}
    elif parameters['MI-method'] == 'approx-linear':
        target = 'mutual_information_estimate'
        args = {'excitation_model': 'lognorm-approx',
                'mutual_information_method': 'polynom'}
    elif parameters['MI-method'] == 'fast':
        target = 'mutual_information_estimate_fast'
        args = {}
    else:
        raise ValueError('Unknown method `%s` for estimating the mutual '
                         'information' % parameters['MI-method'])

    # try finding a library with a non-zero initial mutual information
    model.choose_sensitivity_matrix(**library_opt)
    
    # optimize the interaction matrix
    result = model.optimize_library(target, args=args,
                                    method=parameters['optimization-scheme'],
                                    steps=parameters['steps'],
                                    verbose=not parameters['quite'],
                                    ret_info=parameters['optimization-info'])
    
    # return result data
    data = {'mutual_information': result[0],
            'sensitivity_matrix': result[1],
            'parameters': parameters,
            'init_arguments': model.init_arguments,
            'pickled': pickle.dumps(model),}
    if parameters['optimization-info']:
        data['optimization-info'] = result[2]
    return data



def main():
    """ main program """
    
    # setup the argument parsing
    parser = argparse.ArgumentParser(
         description='Program to optimize receptors for the given parameters. '
                     'Note that most parameters can take multiple values, in '
                     'which case all parameter combinations are computed.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-Ns', nargs='+', type=int, required=True,
                        default=argparse.SUPPRESS, help='number of substrates')
    parser.add_argument('-Nr', nargs='+', type=int, required=True,
                        default=argparse.SUPPRESS, help='number of receptors')
    parser.add_argument('--mixture-scheme', type=str,
                        default='random_uniform',
                        choices=['const', 'linear', 'random_uniform'],
                        help='scheme for picking substrate probabilities')
    parser.add_argument('-m', '--mixture-size', metavar='M', nargs='+',
                        type=float, required=True, default=argparse.SUPPRESS,
                        help='average number of substrates per mixture')
    parser.add_argument('--correlation-scheme', type=str, default='const',
                        choices=['const', 'random_binary', 'random_uniform',
                                 'random_normal'],
                        help='scheme for picking substrate correlations')
    parser.add_argument('--correlation-magnitude', '-corr', metavar='C',
                        nargs='+', type=float, default=[0],
                        help='magnitude of the substrate correlations')
    conc_dists = LibrarySparseNumeric.concentration_distributions
    parser.add_argument('--concentration-distribution', type=str,
                        default='exponential', choices=conc_dists,
                        help='concentration distribution of ligands')
    parser.add_argument('--concentration-scheme', type=str, default='const',
                        choices=['const', 'random_uniform'],
                        help='scheme for picking substrate concentrations')
    parser.add_argument('--concentration-mean', '-conc', metavar='c',
                        nargs='+', type=float, default=[1],
                        help='mean concentration when ligand is present')
    parser.add_argument('--concentration-var', '-conc-var', metavar='v',
                        nargs='+', type=float, default=[1],
                        help='variance of concentration when ligand is present')
    parser.add_argument('--MI-method', type=str, default='numeric',
                        choices=['numeric', 'approx', 'approx-gaussian',
                                 'approx-linear', 'fast'],
                        help='method for estimating the mutual information')
    parser.add_argument('--optimization-scheme', type=str,
                        default='cma',
                        choices=['cma', 'cma-parallel', 'Nelder-Mead', 'BFGS'],
                        help='optimization scheme to use')
    parser.add_argument('--optimization-info', action='store_true',
                        default=False,
                        help='store extra information about the optimization')
    parser.add_argument('-s', '--steps', nargs='+', type=int, default=[100000],
                        help='steps in simulated annealing')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='number of repeats for each parameter set')
    cpus = mp.cpu_count()
    parser.add_argument('-p', '--parallel', action='store', nargs='?',
                        default=1, const=cpus, type=int,
                        help='use multiple processes. %d processes are used if '
                             'only -p is given, without the number.' % cpus)
    parser.add_argument('-q', '--quite', action='store_true',
                        default=False, help='silence the output')
    parser.add_argument('--progress', action='store_true',
                        help='display some progress output', default=False)
    parser.add_argument('-f', '--filename', default='result.pkl',
                        help='filename of the result file')
    
    # fetch the arguments and build the parameter list
    args = parser.parse_args()
    arg_list = (args.Ns, args.Nr, args.mixture_size, args.concentration_mean,
                args.concentration_var, args.correlation_magnitude, args.steps,
                range(args.repeat))

    # determine the number of jobs
    job_count = 1
    for arg in arg_list:
        job_count *= len(arg)
        
    # build a list with all the jobs
    job_list = [{'Ns': Ns, 'Nr': Nr,
                 'mixture-scheme': args.mixture_scheme,
                 'mixture-size': m,  
                 'concentration-distribution': args.concentration_distribution,
                 'concentration-scheme': args.concentration_scheme,
                 'concentration-mean': conc_mean,
                 'concentration-var': conc_var,
                 'correlation-scheme': args.correlation_scheme,
                 'correlation-magnitude': corr,
                 'MI-method': args.MI_method,
                 'optimization-scheme': args.optimization_scheme,
                 'optimization-info': args.optimization_info,
                 'steps': steps,
                 'quite': args.quite,
                 'job_count': job_count, 'progress': args.progress}
                 for Ns, Nr, m, conc_mean, conc_var, corr, steps, _
                    in itertools.product(*arg_list)]

    # do the optimization
    if args.parallel > 1 and len(job_list) > 1:
        results = mp.Pool(args.parallel).map(optimize_library, job_list)
    else:
        results = list(map(optimize_library, job_list))
        
    # write the pickled result to file
    with open(args.filename, 'wb') as fp:
        pickle.dump(results, fp, pickle.HIGHEST_PROTOCOL)
    
    

if __name__ == '__main__':
    main()
    
