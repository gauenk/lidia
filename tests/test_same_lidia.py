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
import n4net
from n4net.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

# -- package imports [to test] --
import lidia
from lidia.model_io import get_lidia_model as get_lidia_model_ntire
from lidia.nl_model_io import get_lidia_model as get_lidia_model_nl
from lidia.data import save_burst

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

#
#
# -- Test original LIDIA v.s. modular (n4net) LIDIA --
#
#

# @pytest.mark.skip()
def test_same_lidia():

    # -- params --
    sigma = 50.
    device = "cuda:0"
    ps = 5
    vid_set = "toy"
    vid_name = "text_tourbus"
    # vid_set = "set8"
    # vid_name = "motorbike"

    # -- video --
    vid_cfg = data_hub.get_video_cfg(vid_set,vid_name)
    clean = data_hub.load_video(vid_cfg)[:3,:,:96,:128]
    clean = th.from_numpy(clean).contiguous().to(device)

    # -- set seed --
    seed = 123
    set_seed(seed)

    # -- over training --
    for train in [True,False]:

        # -- get data --
        noisy = clean + sigma * th.randn_like(clean)
        im_shape = noisy.shape

        # -- lidia exec --
        lidia_model = get_lidia_model_nl(device,im_shape,sigma)
        deno_steps = lidia_model.run_parts(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()

        # -- n4net exec --
        n4_model = n4net.lidia.load_model(sigma)
        deno_n4 = n4_model(noisy,sigma,train=train)
        deno_n4 = deno_n4.detach()

        # -- test --
        error = th.sum((deno_n4 - deno_steps)**2).item()
        assert error < 1e-10
#
#
# -- Test modular (n4net) LIDIA [same as og] v.s. diffy (n4net) LIDIA --
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
    verbose = True
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

        # -- n4net exec --
        n4_model = n4net.lidia.load_model(sigma).to(device)
        deno_steps = n4_model(noisy,sigma,train=train)
        print_gpu_stats(gpu_stats,"post-Step")
        # with th.no_grad():
        #     deno_steps = n4_model(noisy,sigma,train=train)
        deno_steps = deno_steps.detach()/255.

        # -- gpu info --
        print_gpu_stats(gpu_stats,"post-Step.")
        print_peak_gpu_stats(gpu_stats,"post-Step.")

        # -- n4net exec --
        n4b_model = n4net.batched_lidia.load_model(sigma).to(device)
        deno_n4 = n4b_model(noisy,sigma,train=train,batch_size=batch_size)
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
            print("Tain: ",train)
            print("Error: ",error)
            print("Tol: ",tol)
        assert error < tol # allow for batch-norm artifacts

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

    # -- n4net exec --
    set_seed(seed)
    n4_model = n4net.lidia.load_model(sigma).to(device)
    n4_model.run_internal_adapt(noisy,sigma,nsteps=10,nepochs=1)
    deno_n4 = n4_model(noisy,sigma)
    # with th.no_grad():
    #     deno_n4 = n4_model(noisy,sigma)
    deno_n4 = deno_n4.detach()/255.

    # -- n4net exec --
    set_seed(seed)
    n4b_model = n4net.batched_lidia.load_model(sigma).to(device)
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
    assert error < 1e-3

    # -- test --
    error = th.mean((deno_n4 - deno_n4b)**2).item()
    assert error < 1e-3 # allow for batch-norm artifacts
    error = th.max((deno_n4 - deno_n4b)**2).item()
    assert error < 2*1e-2 # allow for batch-norm artifacts

