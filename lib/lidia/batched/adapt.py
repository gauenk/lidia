"""
Functions for internal domain adaptation.

"""

# -- misc --
import sys,math,gc
from .misc import get_default_config,crop_offset

# -- data structs --
import torch.utils.data as data
from lidia.utils.adapt_data import ImagePairDataSet
from lidia.utils.adapt_rpd import RegionProposalData

# -- linalg --
import torch as th
import numpy as np
from einops import repeat,rearrange

# -- path mgmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- separate class and logic --
from lidia.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Run Adaptation of the Network to Image
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


@register_method
def run_internal_adapt(self,_noisy,sigma,srch_img=None,flows=None,
                       clean_gt=None,region_gt=None,verbose=False):
    # -- unpack --
    p = rename_config(self.adapt_cfg)

    # -- manage noisy --
    if verbose: print("Running Internal Adaptation.")
    noisy = (_noisy/255. - 0.5)/0.5
    if not(clean_gt is None): _clean_gt =  (clean_gt/255. - 0.5)/0.5
    else: _clean_gt = None
    opt = get_default_config(sigma)
    if not(srch_img is None):
        _srch_img = (srch_img/255.-0.5)/0.5
        _srch_img = _srch_img.contiguous()
    else: _srch_img = noisy

    for astep in range(p.nadapts):
        with th.no_grad():
            clean_raw = self(noisy,_srch_img,flows=flows,rescale=False)
        clean = clean_raw.detach().clamp(-1, 1)
        psnrs = adapt_step(self, clean, _srch_img, flows, opt,
                           batch_size = p.batch_size,
                           batch_size_te = p.batch_size_te,
                           nsteps = p.nsteps, nepochs = p.nepochs,
                           noise_sim = p.noise_sim,
                           adapt_mtype = p.adapt_mtype,
                           sobel_nlevels = p.sobel_nlevels,
                           region_template = p.region_template,
                           noisy_gt = noisy,clean_gt = _clean_gt,
                           region_gt = region_gt, verbose = verbose)
        return psnrs

@register_method
def run_external_adapt(self,_noisy,_clean,sigma,srch_img=None,
                       flows=None,verbose=False):

    # -- unpack --
    p = rename_config(self.adapt_cfg)

    if verbose: print("Running External Adaptation.")
    # -- setup --
    opt = get_default_config(sigma)
    nadapts = 1
    clean = (_clean/255. - 0.5)/0.5
    # -- adapt --
    if not(srch_img is None):
        _srch_img = srch_img.contiguous()
        _srch_img = (_srch_img/255. - 0.5)/0.5
    else: _srch_img = clean

    # -- eval before --
    # noisy = add_noise_to_image(clean, noise_sim, opt.sigma)
    eval_nl(self,noisy,clean,_srch_img,flows,verbose)

    for astep in range(nadapts):
        adapt_step(self, clean, _srch_img, flows, opt,
                   batch_size = p.batch_size,
                   batch_size_te = p.batch_size_te,
                   noise_sim = p.noise_sim,
                   nsteps=p.nsteps,nepochs=p.nepochs,
                   adapt_mtype = p.adapt_mtype,
                   sobel_nlevels = p.sobel_nlevels,
                   region_template=p.region_template,
                   verbose=verbose)

def rslice(vid,coords):
    if coords is None: return vid
    fs,fe,t,l,b,r = coords
    return vid[fs:fe,:,t:b,l:r]

def compute_psnr(vid_a,vid_b):
    t = vid_a.shape[0]
    mse = th.mean((vid_a.reshape(t,-1)/2. - vid_b.reshape(t,-1)/2.)**2)
    psnr = -10 * th.log10(mse)
    psnr = psnr.cpu().numpy()
    return psnr

def adapt_step(nl_denoiser, clean, srch_img, flows, opt,
               nsteps=100, nepochs=5,
               batch_size=-1, batch_size_te = 390*60,
               noise_sim = None, sobel_nlevels = 3,
               adapt_mtype="default", region_template = "3_96_96",
               noisy_gt=None,clean_gt=None,region_gt=None,verbose=False):

    # -- psnrs --
    psnrs = []
    if not(clean_gt is None):
        print(clean.shape,clean_gt.shape)
        psnr0 = compute_psnr(clean,clean_gt)
        # print(psnr0)
        psnrs.append(psnr0)

    # -- optims --
    criterion = th.nn.MSELoss(reduction='mean')
    optim = th.optim.Adam(nl_denoiser.parameters(), lr=opt.lr,
                              betas=(0.9, 0.999), eps=1e-8)

    # -- get data --
    loader = get_adapt_dataset(clean,adapt_mtype,region_template,sobel_nlevels)

    # -- train --
    noisy = add_noise_to_image(clean, noise_sim, opt.sigma)

    # -- epoch --
    for epoch in range(nepochs):

        # -- info --
        if verbose:
            print('Training epoch {} of {}'.format(epoch + 1, nepochs))

        # -- garbage collect --
        sys.stdout.flush()
        gc.collect()
        th.cuda.empty_cache()

        # -- loaders --
        device = next(nl_denoiser.parameters()).device
        iloader = enumerate(loader)
        nsamples = min(len(loader),nsteps)
        for i, region in iloader:

            # -- tenors on device --
            noisy_i = add_noise_to_image(clean,noise_sim,opt.sigma)

            # -- forward pass --
            optim.zero_grad()
            nl_denoiser.train()
            image_dn = nl_denoiser(noisy_i,srch_img=None,flows=flows,
                                   rescale=False,region=region)

            # -- post-process images --
            image_dn = image_dn.clamp(-1,1)
            clean_r = rslice(clean,region)

            # -- compute loss --
            loss = th.log10(criterion(image_dn/2., clean_r/2.))
            assert not np.isnan(loss.item())

            # -- update step --
            loss.backward()
            optim.step()

            # -- memory dump --
            gc.collect()
            th.cuda.empty_cache()

            # -- logging --
            if (i % 25 == 0) or (nsteps == i):
                nl_denoiser.eval()
                with th.no_grad():
                    deno_gt = nl_denoiser(noisy_gt,srch_img=None,flows=flows,
                                          train=False,rescale=False,
                                          region=region_gt)
                    clean_gt_r = rslice(clean_gt,region_gt)
                    psnr_gt = compute_psnr(deno_gt,clean_gt_r)
                    msg = "[%d/%d] Adaptation update: %2.3f"
                    print(msg % (i,nsteps,psnr_gt))
                    psnrs.append(psnr_gt)

            # -- message --
            if verbose:
                print("Processing [%d/%d]: %2.2f" % (i,nsamples,-10*loss.item()))
            batch_bool = i == nsteps
            epoch_bool = (epoch + 1) % opt.epochs_between_check == 0
            print_bool = batch_bool and epoch_bool
            if print_bool:
                gc.collect()
                th.cuda.empty_cache()
                nl_denoiser.eval()
                with th.no_grad():
                    deno = nl_denoiser(noisy,srch_img.clone(),flows,
                                       rescale=False)
                deno = deno.detach().clamp(-1, 1)
                mse = criterion(deno / 2,clean / 2).item()
                train_psnr = -10 * math.log10(mse)
                psnrs.append(train_psnr)
                if verbose:
                    a,b,c = epoch + 1, nepochs, train_psnr
                    msg = 'Epoch {} of {} done, training PSNR = {:.2f}'.format(a,b,c)
                    print(msg)
                    sys.stdout.flush()
            if i > nsteps: break

    return psnrs


def eval_nl(nl_denoiser,noisy,clean,srch_img,flows,verbose=True):
    deno = nl_denoiser(noisy,srch_img.clone(),flows=flows,rescale=False)
    deno = deno.detach().clamp(-1, 1)
    mse = th.mean((deno / 2-clean / 2)**2).item()
    psnr = -10 * math.log10(mse)
    msg = 'PSNR = {:.2f}'.format(psnr)
    if verbose:
        print(msg)

def get_adapt_dataset(clean,mtype,region_template,nlevels=3):
    rpn = RegionProposalData(clean,mtype,region_template,nlevels)
    return rpn

def add_noise_to_image(clean, noise_sim, sigma):
    if noise_sim is None:
        noisy = clean + sigma_255_to_torch(sigma) * th.randn_like(clean)
    else:
        with th.no_grad():
            noisy = noise_sim(clean)
    return noisy

def sigma_255_to_torch(sigma_255):
    return (sigma_255 / 255) / 0.5

def rename_config(_cfg):
    cfg = edict()
    pairs = {'internal_adapt_nsteps':"nsteps",
             'internal_adapt_nepochs':"nepochs",
             'internal_adapt_nadapts':"nadapts",
             "bs":"batch_size","bs_te":"batch_size_te",
             "adapt_noise_sim":"noise_sim",
             'adapt_mtype':"adapt_mtype",
             'adapt_region_template':"region_template",
             'sobel_nlevels':"sobel_nlevels"}
    for key0,key1 in pairs.items():
        cfg[key1] = _cfg[key0]
    return cfg

