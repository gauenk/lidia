# -- Remove Numba Warnings --
import warnings
from numba import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- api --
from . import original
from . import refactored
from . import batched
from . import testing
from . import plots
