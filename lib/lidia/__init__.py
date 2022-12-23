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
from . import flow
from .batched import extract_model_config

# -- paper --
from . import aaai23

# -- api for searching --
from . import search
from .search import get_search,extract_search_config

# -- model api --
from .utils import optional

def load_model(cfg):
    mtype = optional(cfg,'model_type','batched')
    if mtype == "original":
        model = original.load_model(cfg)
    elif mtype == "batched":
        model = batched.load_model(cfg)
    elif mtype == "refactored":
        model = refactored.load_model(cfg)
    else:
        raise ValueError(f"Uknown model [{mtype}]")
    return model


def load_model_old(mtype,sigma,rtype="original"):
    if mtype == "original":
        model = original.load_model(sigma)
    elif mtype == "batched":
        model = batched.load_model(sigma)
    elif mtype == "refactored":
        model = refactored.load_model(sigma,rtype)
    else:
        raise ValueError(f"Uknown model [{mtype}]")
    return model
