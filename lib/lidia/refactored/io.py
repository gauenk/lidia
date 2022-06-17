
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .lidia_structs import LIDIA,OriginalLIDIA,ArchitectureOptions

# -- misc imports --
from .misc import get_default_config,calc_padding,select_sigma

def load_model(sigma,mtype="lidia"):

    # -- get cfg --
    cfg = get_default_config(sigma)
    arch_cfg = ArchitectureOptions(True)

    # -- init model --
    pad_offs, total_pad = calc_padding(arch_cfg)
    if mtype == "lidia":
        nl_denoiser = LIDIA(pad_offs, arch_cfg).to(cfg.device)
    elif mtype == "original":
        nl_denoiser = OriginalLIDIA(pad_offs, arch_cfg).to(cfg.device)
    nl_denoiser.cuda()

    # -- load weights --
    lidia_sigma = select_sigma(sigma)
    fdir = Path(__file__).absolute().parents[0] / "../../../"
    state_fn = fdir / "models/model_state_sigma_{}_c.pt".format(lidia_sigma)
    assert os.path.isfile(str(state_fn))
    model_state = th.load(str(state_fn))
    # state_fn0 = '/home/gauenk/Documents/packages/lidia/lidia-deno/models/model_state_sigma_{}_c.pt'.format(lidia_sigma)
    modded_dict(model_state['state_dict'])
    nl_denoiser.pdn.load_state_dict(model_state['state_dict'])

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

