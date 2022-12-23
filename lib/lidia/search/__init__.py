"""

Interface to compare search methods

"""

# -- impl objs --
from .lnl import LNLSearch
from .nl import NLSearch

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_search_config = partial(extract_config,_fields)

def get_search(cfg):

    # -- unpack --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    name = optional(cfg,'name',"lnl")
    k = optional(cfg,'k',15)
    ps = optional(cfg,'ps',7)
    ws = optional(cfg,'ws',29)
    wt = optional(cfg,'wt',0)
    stride0 = optional(cfg,'stride0',1)
    stride1 = optional(cfg,'stride1',1)
    train = optional(cfg,'train',False)
    lidia_pad = optional(cfg,'lidia_pad',True)
    if init: return

    # -- init --
    if name in ["lnl","original"]:
        return LNLSearch(k,ps,ws,stride0,train,lidia_pad)
    elif name in ["nl","ours"]:
        return NLSearch(k,ps,ws,wt,stride0,stride1)
    else:
        raise ValueError(f"Uknown search method [{name}]")

# -- fill fields --
get_search({"__init":True})
