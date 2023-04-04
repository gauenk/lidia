
# -- misc --
import sys,os,copy
from pathlib import Path
from easydict import EasyDict as edict

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .modules import NonLocalDenoiser,ArchitectureOptions

def load_model(cfg):#sigma,rgb=True,device="cuda:0"):

    # -- get cfg --
    sigma = cfg.sigma
    rgb = True
    device="cuda:0"
    cfg = get_default_config(cfg.sigma)
    arch_opt = ArchitectureOptions(rgb=rgb, small_network=False)
    ps = 5 if rgb else 7
    pad_offs, _ = calc_padding(ps)#arch_opt)
    nl_denoiser = NonLocalDenoiser(pad_offs, arch_opt).to(device)

    # -- load weights --
    lidia_sigma = select_sigma(sigma)
    fdir = Path(__file__).absolute().parents[0] / "../../../"
    state_fn = fdir / "models/model_state_sigma_{}_c.pt".format(lidia_sigma)
    assert os.path.isfile(str(state_fn))
    model_state = th.load(str(state_fn))
    # modded_dict(model_state['state_dict'])
    nl_denoiser.patch_denoise_net.load_state_dict(model_state['state_dict'])

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

def get_default_config(sigma):
    cfg = edict()
    cfg.sigma = sigma
    cfg.seed = 123
    cfg.block_w = 64
    cfg.lr = 1e-3
    cfg.epoch_num = 2
    cfg.epochs_between_check = 1
    cfg.max_batches = 500
    cfg.dset_stride = 1
    cfg.train_batch_size = 4
    cfg.device = "cuda:0"
    return cfg

def calc_padding(patch_w=5,k=14):
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + (k-1)
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + k * 2
    offs = total_pad - total_pad0
    return offs,total_pad

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

