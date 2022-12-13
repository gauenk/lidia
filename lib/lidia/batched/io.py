
# -- misc --
import sys,os,copy
from pathlib import Path
from functools import partial

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .lidia_structs import BatchedLIDIA,ArchitectureOptions

# -- misc imports --
from .misc import get_default_config,calc_padding,select_sigma

# -- model io --
from dev_basics import arch_io

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_model_config = partial(extract_config,_fields)

def load_model(cfg):

    # -- allows for all keys to be aggregated at init --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # sigma,name="",lidia_pad=False,match_bn=False,remove_bn=False):

    # -- config --
    device = optional(cfg,"device","cuda:0")
    name = optional(cfg,"name","default")
    sigma = optional(cfg,'sigma',30)
    lidia_pad = optional(cfg,'lidia_pad',False)
    match_bn = optional(cfg,'match_bn',False)
    remove_bn = optional(cfg,'remove_bn',False)
    ps = optional(cfg,'ps',5)
    ws = optional(cfg,'ws',29)
    wt = optional(cfg,'wt',0)
    bs = optional(cfg,'bs',0)
    bs_te = optional(cfg,'bs_te',390*90)
    bs_alpha = optional(cfg,'bs_alpha',0.25)
    adapt_cfg = extract_adapt_config(cfg,optional)
    io_cfg = extract_io_config(cfg,optional)
    if init: return

    # -- get arch cfg --
    # cfg = get_default_config(sigma)
    arch_cfg = ArchitectureOptions(True)

    # -- init model --
    pad_offs, total_pad = calc_padding()#arch_cfg)
    nl_denoiser = BatchedLIDIA(adapt_cfg, pad_offs, arch_cfg,
                               name = name,
                               lidia_pad = lidia_pad,
                               match_bn = match_bn,
                               remove_bn = remove_bn,
                               bs=bs,bs_te=bs_te,ws=ws,wt=wt,
                               bs_alpha=bs_alpha)
    nl_denoiser = nl_denoiser.to(device)
    nl_denoiser.cuda()

    # -- load weights --
    modifier = partial(modded_dict,remove_bn=remove_bn)
    if io_cfg.pretrained_load:
        arch_io.load_checkpoint(nl_denoiser.pdn,io_cfg.pretrained_path,
                                io_cfg.pretrained_root,io_cfg.pretrained_type,
                                modifier)

    # -- old api --
    # lidia_sigma = select_sigma(sigma)
    # fdir = Path(__file__).absolute().parents[0] / "../../../"
    # state_fn = fdir / "models/model_state_sigma_{}_c.pt".format(lidia_sigma)

    return nl_denoiser

def extract_adapt_config(_cfg,optional):
    pairs = {'internal_adapt_nsteps':200,
             'internal_adapt_nepochs':1,
             'internal_adapt_nadapts':1,
             'adapt_noise_sim':None,
             'adapt_mtype':"rand",
             'adapt_region_template':"3_128_128",
             'sobel_nlevels':3,
             "bs":32*1024,"bs_te":390*90}
    return extract_pairs(pairs,_cfg,optional)

def extract_io_config(_cfg,optional):
    # lidia_sigma = select_sigma(sigma)
    # fdir = Path(__file__).absolute().parents[0] / "../../../"
    # state_fn = fdir / "models/model_state_sigma_{}_c.pt".format(lidia_sigma)
    pairs = {"pretrained_load":True,
             "pretrained_path":"",
             "pretrained_type":"mod",
             "pretrained_root":"."}
    return extract_pairs(pairs,_cfg,optional)


load_model({"__init":True}) # -- fill fields --

def modded_dict(mdict,remove_bn=None):
    names = list(mdict.keys())
    for name in names:
        name_og = copy.copy(name)
        name = name.replace("separable_fc_net","sep_net")
        name = name.replace("ver_hor_agg0_pre","agg0_pre")
        name = name.replace("ver_hor_bn_re_agg0_post","agg0_post")
        name = name.replace("ver_hor_agg1_pre","agg1_pre")
        name = name.replace("ver_hor_bn_re_agg1_post","agg1_post")
        value = mdict[name_og]
        del mdict[name_og]
        mdict[name] = value
        if remove_bn:
            if ".bn" in name_og:
                del mdict[name]
