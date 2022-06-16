
# -- misc --
import sys,os,copy

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .lidia_structs import LIDIA,ArchitectureOptions

# -- misc imports --
from .misc import get_default_config,calc_padding,select_sigma

def load_model(sigma):

    # -- get cfg --
    cfg = get_default_config(sigma)
    arch_cfg = ArchitectureOptions(True)

    # -- init model --
    pad_offs, total_pad = calc_padding(arch_cfg)
    nl_denoiser = LIDIA(pad_offs, arch_cfg).to(cfg.device)
    nl_denoiser.cuda()

    # -- load weights --
    lidia_sigma = select_sigma(sigma)
    state_fn0 = '/home/gauenk/Documents/packages/lidia/lidia-deno/models/model_state_sigma_{}_c.pt'.format(lidia_sigma)
    assert os.path.isfile(state_fn0)
    model_state0 = th.load(state_fn0)
    modded_dict(model_state0['state_dict'])
    nl_denoiser.pdn.load_state_dict(model_state0['state_dict'])

    return nl_denoiser

def modded_dict(mdict):
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

