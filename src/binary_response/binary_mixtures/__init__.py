from .lib_bin_numeric import LibraryBinaryNumeric
from .lib_bin_theory import LibraryBinaryUniform

# provide deprecated classes for compatibility
from utils.misc import DeprecationHelper
ReceptorLibraryNumeric = DeprecationHelper(LibraryBinaryNumeric)
ReceptorLibraryUniform = DeprecationHelper(LibraryBinaryUniform)

__all__ = ['LibraryBinaryNumeric', 'LibraryBinaryUniform',
           'ReceptorLibraryNumeric', 'ReceptorLibraryUniform']

# try importing numba for speeding up calculations
try:
    from .numba_speedup import numba_patcher
    numba_patcher.enable() #< enable the speed-up by default
except ImportError:
    import logging
    logging.warning('Numba could not be loaded. Slow functions will be used')
    numba_patcher = None
    