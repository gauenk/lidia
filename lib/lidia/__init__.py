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
from . import configs
from . import explore_configs

# -- paper --
from . import aaai23

def load_model(mtype,sigma,rtype="original"):
    if mtype == "original":
        model = original.load_model(sigma)
    elif mtype == "batched":
        model = batched.load_model(sigma)
    elif mtype == "refactored":
        model = refactored.load_model(sigma,rtype)
    else:
        raise ValueError(f"Uknown model [{mtype}]")
    return model
