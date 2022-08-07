"""

When first created our model is identical to lidia

"""

# -- misc --
import sys,tqdm,pytest,math,random
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
import dnls # supporting
import lidia
from lidia.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import lidia
# from lidia.model_io import get_lidia_model as get_lidia_model_ntire
# from lidia.nl_model_io import get_lidia_model as get_lidia_model_nl
# from lidia.data import save_burst

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_same_lidia/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)
def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

#
#
# -- Test original LIDIA v.s. modular (lidia) LIDIA --
#
#

# @pytest.mark.skip()
def test_same_original_refactored():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    # vid_set = "set8"
    # vid_name = "motorbike"
    verbose = False

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:96,:128]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- set seed --
    seed = 123
    # set_seed(seed)

    # -- over training --
    for train in [False,True]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape

        # -- lidia exec --
        lidia_model = lidia.original.load_model(sigma)
        with th.no_grad():
            deno_lid = lidia_model(noisy,train=train,normalize=True).detach()

        # -- lidia exec --
        n4_model = lidia.refactored.load_model(sigma,"original")
        with th.no_grad():
            deno_ref = n4_model(noisy,sigma,train=train).detach()

        # -- test --
        error = th.sum((deno_lid - deno_ref)**2).item()
        if verbose: print("error: ",error)
        assert error < 1e-15

# @pytest.mark.skip()
def test_same_refactored_batched():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    # vid_set = "set8"
    # vid_name = "motorbike"
    verbose = False

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:96,:128]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- set seed --
    seed = 123
    set_seed(seed)

    # -- over training --
    for train in [False,True]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape

        # -- lidia exec --
        lidia_model = lidia.batched.load_model(sigma,lidia_pad=True,
                                               match_bn=True)
        with th.no_grad():
            deno_lid = lidia_model(noisy,sigma,train=train).detach()

        # -- lidia exec --
        n4_model = lidia.refactored.load_model(sigma)
        with th.no_grad():
            deno_ref = n4_model(noisy,sigma,train=train).detach()

        # -- max_diff --
        diff = th.abs(deno_lid - deno_ref)
        dmax = diff.max()
        # diff /= diff.max()
        # lidia.testing.data.save_burst(diff,"./output/tests/","diff")
        if verbose:
            print("Max L1: ",dmax)
        assert dmax < 1e-4

        # -- test --
        error = th.sum((deno_lid - deno_ref)**2).item()
        if verbose:
            print("Error: ",error)
        assert error < 1e-4

#
#
# -- Test modular (lidia) LIDIA [same as og] v.s. diffy (lidia) LIDIA --
#
#

# @pytest.mark.skip()
def test_batched():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    # vid_set = "set8"
    # vid_name = "motorbike"
    vid_set = "toy"
    vid_name = "text_tourbus"
    gpu_stats = False
    verbose = False
    batch_size = 1024
    th.cuda.set_device(0)

    # -- set seed --
    seed = 123
    set_seed(seed)

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:96,:156]
    # clean = data_hub.load_video(vid_cfg)[:3,:,:96,:96]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- gpu info --
    print_peak_gpu_stats(gpu_stats,"Init.")

    # -- over training --
    for train in [False,True]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape
        noisy = noisy.contiguous()

        # -- lidia exec --
        n4_model = lidia.refactored.load_model(sigma).to(device)
        deno_steps = n4_model(noisy,sigma,train=train)
        print_gpu_stats(gpu_stats,"post-Step")
        # with th.no_grad():
        #     deno_steps = n4_model(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()/255.

        # -- gpu info --
        print_gpu_stats(gpu_stats,"post-Step.")
        print_peak_gpu_stats(gpu_stats,"post-Step.")

        # -- lidia exec --
        n4b_model = lidia.batched.load_model(sigma,lidia_pad=True).to(device)
        deno_n4 = n4b_model(noisy,sigma,train=train,
                            batch_size=batch_size)
        print_gpu_stats(gpu_stats,"post-Batched.")
        print_peak_gpu_stats(gpu_stats,"post-Batched.")
        # with th.no_grad():
        #     deno_n4 = n4b_model(noisy,sigma,train=train)
        deno_n4 = deno_n4.detach()/255.

        # -- save --
        dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"batched")
        dnls.testing.data.save_burst(deno_steps,SAVE_DIR,"ref")
        diff = th.abs(deno_steps - deno_n4)
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- test mse --
        error = th.mean((deno_n4 - deno_steps)**2).item()
        if train:
            if batch_size == -1:
                tol = 1e-14
            else:
                tol = 1e-3 # batch effects
        else: # different on the edge
            if batch_size == -1:
                tol = 1e-4
            else:
                tol = 1e-3 # batch effects
        if verbose:
            print("Train: ",train)
            print("Error: ",error)
            print("Tol: ",tol)
        assert error < tol # allow for batch-norm artifacts

        # -- gpu info --
        print_gpu_stats(gpu_stats,"final.")
        print_peak_gpu_stats(gpu_stats,"final.")

#
#
# -- Test denoising on only inset region --
#
#

# @pytest.mark.skip()
def test_inset_deno():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    # vid_set = "set8"
    # vid_name = "motorbike"
    vid_set = "toy"
    vid_name = "text_tourbus"
    gpu_stats = False
    verbose = False
    batch_size = -1#128*128
    remove_bn = False
    th.cuda.set_device(0)

    # -- set seed --
    seed = 123
    set_seed(seed)

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:256,:256]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- inset region --
    coords = [0,3,2,2,124,124]#32,32]
    def cslice(vid,coords):
        fs,fe,t,l,b,r = coords
        return vid[fs:fe,:,t:b,l:r]
    csize = (100,100)

    # -- gpu info --
    print_peak_gpu_stats(gpu_stats,"Init.")

    # -- over training --
    for train in [True,False]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape
        noisy = noisy.contiguous()

        # -- lidia exec --
        b_model = lidia.batched.load_model(sigma,name="first").to(device)
        deno_b = b_model(noisy.clone(),sigma,train=train,batch_size=batch_size)
        print_gpu_stats(gpu_stats,"post-step")
        deno_b = deno_b.detach()/255.
        deno_b = cslice(deno_b,coords)
        deno_b = center_crop(deno_b,csize)

        # -- gpu info --
        print_gpu_stats(gpu_stats,"post-step.")
        print_peak_gpu_stats(gpu_stats,"post-step.")

        # -- lidia exec --
        n4b_model = lidia.batched.load_model(sigma,name="second").to(device)
        deno_n4 = n4b_model(noisy.clone(),sigma,train=train,
                            batch_size=batch_size,region=coords)
        deno_n4 = deno_n4.detach()/255.
        deno_n4 = center_crop(deno_n4,csize)
        print_gpu_stats(gpu_stats,"post-batched.")
        print_peak_gpu_stats(gpu_stats,"post-batched.")

        # -- save --
        dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"batched")
        dnls.testing.data.save_burst(deno_b,SAVE_DIR,"ref")
        diff = th.abs(deno_b - deno_n4)
        dmax = diff.max().item()
        diff /= diff.max()
        dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

        # -- test mse --
        error = th.sum((deno_n4 - deno_b)**2).item()
        if verbose:
            print("Train: ",train)
            print("L1-Max: ",dmax)
            print("Error: ",error)
        assert dmax < 5.e-3 # allow error from BN
        assert error < 5.e-2 # allow error from BN

        # -- gpu info --
        print_gpu_stats(gpu_stats,"final.")
        print_peak_gpu_stats(gpu_stats,"final.")


#
#
# -- Test internal adaptations for LIDIA and BatchedLIDIA --
#
#

# @pytest.mark.skip()
def test_internal_adapt():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    gpu_stats = False
    seed = 123
    verbose = False
    th.cuda.set_device(0)

    # -- set seed --
    set_seed(seed)

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:128,:128]
    # clean = data_hub.load_video(vid_cfg)[:3,:,:96,:96]
    clean = th.from_numpy(clean).contiguous().to(device)
    clean_01 = clean/255.


    # -- get data --
    noisy = clean + sigma * th.randn_like(clean)
    im_shape = noisy.shape
    noisy = noisy.contiguous()

    # -- lidia refactored exec --
    set_seed(seed)
    n4_model = lidia.refactored.load_model(sigma).to(device)
    n4_model.run_internal_adapt(noisy,sigma,nsteps=10,nepochs=1)
    deno_n4 = n4_model(noisy,sigma)
    # with th.no_grad():
    #     deno_n4 = n4_model(noisy,sigma)
    deno_n4 = deno_n4.detach()/255.

    # -- lidia batched exec --
    set_seed(seed)
    n4b_model = lidia.batched.load_model(sigma).to(device)
    n4b_model.run_internal_adapt(noisy,sigma,nsteps=10,nepochs=1)
    deno_n4b = n4b_model(noisy,sigma)
    # with th.no_grad():
    #     deno_n4 = n4b_model(noisy,sigma,train=train)
    deno_n4b = deno_n4b.detach()/255.

    # -- save --
    dnls.testing.data.save_burst(deno_n4,SAVE_DIR,"ref")
    dnls.testing.data.save_burst(deno_n4b,SAVE_DIR,"batched")
    diff = th.abs(deno_n4 - deno_n4b)
    diff /= diff.max()
    dnls.testing.data.save_burst(diff,SAVE_DIR,"diff")

    # -- psnrs --
    mse_n4 = th.mean((deno_n4 - clean_01)**2).item()
    psnr_n4 = -10 * math.log10(mse_n4)
    mse_n4b = th.mean((deno_n4b - clean_01)**2).item()
    psnr_n4b = -10 * math.log10(mse_n4b)
    if verbose:
        print("PSNR[stnd]: %2.3f" % psnr_n4)
        print("PSNR[batched]: %2.3f" % psnr_n4b)
    error = np.sum((psnr_n4 - psnr_n4b)**2).item()
    assert error < 5e-3

    # -- test --
    error = th.mean((deno_n4 - deno_n4b)**2).item()
    assert error < 1e-3 # allow for batch-norm artifacts
    error = th.max((deno_n4 - deno_n4b)**2).item()
    assert error < 2*1e-2 # allow for batch-norm artifacts

