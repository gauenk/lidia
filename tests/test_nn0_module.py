
# -- misc --
import sys,tqdm
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import lidia
from lidia.testing.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)


def run_rgb2gray(tensor):
    kernel = th.tensor([0.2989, 0.5870, 0.1140], dtype=th.float32)
    kernel = kernel.view(1, 3, 1, 1)
    rgb2gray = th.nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1, 1),bias=False)
    rgb2gray.weight.data = kernel
    rgb2gray.weight.requires_grad = False
    rgb2gray = rgb2gray.to(tensor.device)
    tensor = rgb2gray(tensor)
    return tensor

def run_rgb2gray_patches(patches,ps):
    t,h,w,k,d = patches.shape
    patches = rearrange(patches,'t h w k (c ph pw) -> (t h w k) c ph pw',ph=ps,pw=ps)
    patches = run_rgb2gray(patches)
    patches = rearrange(patches,'(t h w k) 1 ph pw -> t h w k (ph pw)',t=t,h=h,w=w)
    return patches

#
#
# -- Primary Testing Class --
#
#

class TestNn0(unittest.TestCase):

    #
    # -- Load Data --
    #

    def load_burst(self,name,ext="jpg",device="cuda:0"):
        # -- video --
        vid_set = "toy"
        vid_name = "text_tourbus"
        vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
        clean = data_hub.load_video(vid_cfg)[:3,:,:96,:128]
        clean = th.from_numpy(clean).contiguous().to(device)
        return clean

    def test_nonlocal0(self):
        # -- params --
        # name = "davis_baseball_64x64"
        name = "davis_salsa"
        sigma = 50.
        device = "cuda:0"

        # -- set seed --
        seed = 123
        th.manual_seed(seed)
        np.random.seed(seed)

        # -- exec --
        self.compare_nn0(name,sigma,False,device)
        self.compare_nn0(name,sigma,True,device)

    def compare_nn0(self,name,sigma,train,device):

        # -- get data --
        clean = self.load_burst(name).to(device)[:5,:,:96,:128]
        noisy = clean + sigma * th.randn_like(clean)
        t,c,h,w = clean.shape
        im_shape = noisy.shape
        ps = 5

        # -- load model --
        model_ntire = lidia.refactored.load_model(sigma,"original")
        model_nl = lidia.refactored.load_model(sigma,"lidia")

        # -- exec nl search  --
        nl_output = model_nl.run_nn0(noisy/255.,train=train)
        nl_patches = nl_output[0]
        nl_dists = nl_output[1]
        nl_inds = nl_output[2]

        # -- exec ntire search  --
        ntire_output = model_ntire.run_nn0(noisy/255.,train=train)
        ntire_patches = ntire_output[0]
        ntire_dists = ntire_output[1]
        ntire_inds = ntire_output[2]

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #        Viz
        #
        # -=-=-=-=-=-=-=-=-=-=-=-


        # -- error --
        # error = th.abs(ntire_patches - nl_patches)
        # args = th.where(error > 1e-2)
        # print(args)
        # ti = args[0][0].item()
        # hi = args[1][0].item()
        # wi = args[2][0].item()
        # ki = args[3][0].item()
        # print(ti,hi,wi,ki)
        # print(ntire_dists[ti,hi,wi])
        # print(nl_dists[ti,hi,wi])


        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        #     Comparisons
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        """
        We can't do direct comparisons because equal dist
        locations may be swapped.
        """

        # -- dists  --
        error = (ntire_dists - nl_dists)**2
        error = error.sum().item()
        assert error < 1e-8

        # -- ave patch content  --
        # allow for some error b/c k^th rank may have multi. equiv dists
        nl_mp = nl_patches[...,:,:].mean(-2)
        ntire_mp = ntire_patches[...,:,:].mean(-2)
        error = (nl_mp - ntire_mp)**2
        error = error.sum().item()
        assert error < 10.

        # -- 1st patch content  --
        error = (nl_patches[...,0,:] - ntire_patches[...,0,:])**2
        error = error.sum().item()
        assert error < 1e-6

        # -- [nl] patch-based dists == dist --
        nl_patches = run_rgb2gray_patches(nl_patches,ps)
        dists = (nl_patches - nl_patches[...,[0],:])**2
        nl_pdists = th.sum(dists,-1)
        error = (nl_pdists[...,1:] - nl_dists[...,:-1])**2
        error = error.sum().item()
        assert error < 1e-6

        # -- [ntire] patch-based dists == dist --
        ntire_patches = run_rgb2gray_patches(ntire_patches,ps)
        dists = (ntire_patches - ntire_patches[...,[0],:])**2
        ntire_pdists = th.sum(dists,-1)
        error = (ntire_pdists[...,1:] - ntire_dists[...,:-1])**2
        error = error.sum().item()
        assert error < 1e-6

