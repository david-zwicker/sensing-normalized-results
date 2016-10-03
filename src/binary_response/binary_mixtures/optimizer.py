'''
Created on May 14, 2015

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import functools
import random
import time

from simanneal import Annealer


   
class ReceptorOptimizerAnnealer(Annealer):
    """ class that manages the simulated annealing """
    updates = 20    # Number of outputs
    copy_strategy = 'method'


    def __init__(self, model, target, direction='max', args=None,
                 ret_info=False, values_step=1):
        """ initialize the optimizer with a `model` to run and a `target`
        function to call. """
        if ret_info:
            self.info = {'values': {}}
            self.values_step = values_step
            self._step = 0
        else:
            self.info = None
        self.model = model

        target_function = getattr(model, target)
        if args is not None:
            self.target_func = functools.partial(target_function, **args)
        else:
            self.target_func = target_function

        assert direction in ('min', 'max')
        self.direction = direction
        super(ReceptorOptimizerAnnealer, self).__init__(model.sens_mat)
   
   
    def move(self):
        """ change a single entry in the interaction matrix """   
        i = random.randrange(self.state.size)
        self.state.flat[i] = 1 - self.state.flat[i]

      
    def energy(self):
        """ returns the energy of the current state """
        self.model.sens_mat = self.state
        value = self.target_func()
        
        if self.info is not None and self._step % self.values_step == 0:
            self.info['values'][self._step] = value
            self._step += 1
        
        if self.direction == 'max':
            return -value
        else:
            return value
    

    def optimize(self):
        """ optimizes the receptors and returns the best receptor set together
        with the achieved mutual information """
        state_best, value_best = self.anneal()
        if self.info is not None:
            self.info['total_time'] = time.time() - self.start    
            self.info['states_considered'] = self.steps
            self.info['performance'] = self.steps / self.info['total_time']
        
        if self.direction == 'max':
            return -value_best, state_best
        else:
            return value_best, state_best

