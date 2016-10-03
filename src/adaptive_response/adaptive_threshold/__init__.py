from .at_numeric import AdaptiveThresholdNumeric
from .at_theory import (AdaptiveThresholdTheory,
                        AdaptiveThresholdTheoryReceptorFactors)

__all__ = ['AdaptiveThresholdNumeric', 'AdaptiveThresholdTheory',
           'AdaptiveThresholdTheoryReceptorFactors']


# try importing numba for speeding up calculations
try:
    from .numba_speedup import numba_patcher
    numba_patcher.enable() #< enable the speed-up by default
except ImportError:
    import logging
    logging.warning('Numba could not be loaded. Slow functions will be used')
    numba_patcher = None