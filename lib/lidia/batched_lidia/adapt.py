"""
Functions for internal domain adaptation.

"""

# -- misc --
import sys,math,gc
from .misc import get_default_config,crop_offset

# -- data structs --
import torch.utils.data as data
from n4net.utils.adapt_data import ImagePairDataSet

# -- linalg --
import torch as th
import numpy as np
from einops import repeat,rearrange

# -- path mgmnt --
from pathlib import Path

# -- separate class and logic --
from n4net.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Run Adaptation of the Network to Image
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_internal_adapt(self,_noisy,sigma,srch_img=None,flows=None,ws=29,wt=0,
                       batch_size = -1, nsteps=100, nepochs=5, verbose=False):
    if verbose: print("Running Internal Adaptation.")
    noisy = (_noisy/255. - 0.5)/0.5
    opt = get_default_config(sigma)
    total_pad = 20
    nadapts = 1
    if not(srch_img is None):
        _srch_img = (srch_img/255.-0.5)/0.5
        _srch_img = _srch_img.contiguous()
    else: _srch_img = noisy

    for astep in range(nadapts):
        with th.no_grad():
            clean = self(noisy,sigma,_srch_img,flows=flows,rescale=False,
                         ws=ws,wt=wt,batch_size=batch_size)
        clean = clean.detach().clamp(-1, 1)
        nl_denoiser = adapt_step(self, clean, _srch_img, flows, opt,
                                 total_pad, ws=ws, wt=wt, batch_size=batch_size,
                                 nsteps=nsteps,nepochs=nepochs,verbose=verbose)

@register_method
def run_external_adapt(self,_clean,sigma,srch_img=None,flows=None,ws=29,wt=0,
                       batch_size = -1, nsteps=100, nepochs=5, verbose=False):

    if verbose: print("Running External Adaptation.")
    # -- setup --
    opt = get_default_config(sigma)
    total_pad = 10
    nadapts = 1
    clean = (_clean/255. - 0.5)/0.5
    # -- adapt --
    if not(srch_img is None):
        _srch_img = srch_img.contiguous()
        _srch_img = (_srch_img/255. - 0.5)/0.5
    else: _srch_img = clean

    # -- eval before --
    noisy = add_noise_to_image(clean, opt.sigma)
    eval_nl(self,noisy,clean,_srch_img,flows,opt.sigma,verbose)

    for astep in range(nadapts):
        nl_denoiser = adapt_step(self, clean, _srch_img, flows, opt,
                                 total_pad, ws=ws,wt=wt, batch_size=batch_size,
                                 nsteps=nsteps,nepochs=nepochs,verbose=verbose)

def adapt_step(nl_denoiser, clean, srch_img, flows, opt, total_pad,
               ws=29, wt=0, nsteps=100, nepochs=5, batch_size=-1, verbose=False):

    # -- optims --
    criterion = th.nn.MSELoss(reduction='mean')
    optim = th.optim.Adam(nl_denoiser.parameters(), lr=opt.lr,
                              betas=(0.9, 0.999), eps=1e-8)

    # -- get data --
    loader,batch_last_it = get_adapt_dataset(clean,srch_img,opt,total_pad)

    # -- train --
    noisy = add_noise_to_image(clean, opt.sigma)

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
        for i, (clean_i, srch_i) in iloader:

            # -- tenors on device --
            srch_i = srch_i.to(device=device).contiguous()
            clean_i = clean_i.to(device=device).contiguous()
            noisy_i = clean_i + sigma_255_to_torch(opt.sigma) * th.randn_like(clean_i)
            noisy_i = noisy_i.contiguous()

            # -- forward pass --
            optim.zero_grad()
            image_dn = nl_denoiser(noisy_i,opt.sigma,srch_i,flows=flows,
                                   ws=ws,wt=wt,train=True,rescale=False,
                                   batch_size=batch_size)

            # -- post-process images --
            image_dn = image_dn.clamp(-1,1)
            total_pad = (clean_i.shape[-1] - image_dn.shape[-1]) // 2
            image_ref = crop_offset(clean_i, (total_pad,), (total_pad,))

            # -- compute loss --
            loss = th.log10(criterion(image_dn/2., image_ref/2.))
            assert not np.isnan(loss.item())

            # -- update step --
            loss.backward()
            optim.step()

            if verbose:
                print("Processing [%d/%d]: %2.2f" % (i,nsamples,-10*loss.item()))

            batch_bool = i == batch_last_it
            epoch_bool = (epoch + 1) % opt.epochs_between_check == 0
            print_bool = batch_bool and epoch_bool and verbose
            if print_bool:
                gc.collect()
                th.cuda.empty_cache()
                deno = nl_denoiser(noisy,opt.sigma,srch_img.clone(),flows,
                                   rescale=False,ws=ws,wt=wt)
                deno = deno.detach().clamp(-1, 1)
                mse = criterion(deno / 2,clean / 2).item()
                train_psnr = -10 * math.log10(mse)
                a,b,c = epoch + 1, nepochs, train_psnr
                msg = 'Epoch {} of {} done, training PSNR = {:.2f}'.format(a,b,c)
                print(msg)
                sys.stdout.flush()
            if i > nsteps: break

    return nl_denoiser


def eval_nl(nl_denoiser,noisy,clean,srch_img,flows,sigma,ws=29,wt=0,verbose=True):
    deno = nl_denoiser(noisy,sigma,srch_img.clone(),flows=flows,
                       rescale=False,ws=ws,wt=wt)
    deno = deno.detach().clamp(-1, 1)
    mse = th.mean((deno / 2-clean / 2)**2).item()
    psnr = -10 * math.log10(mse)
    msg = 'PSNR = {:.2f}'.format(psnr)
    if verbose:
        print(msg)


def get_adapt_dataset(clean,srch_img,opt,total_pad):

    # -- prepare data --
    block_w_pad = opt.block_w + 2 * total_pad
    ref_img = clean
    srch_img = srch_img

    # -- create dataset --
    dset = ImagePairDataSet(block_w=block_w_pad,
                            images_a=ref_img, images_b=srch_img,
                            stride=opt.dset_stride)

    # -- create loader --
    loader = data.DataLoader(dset,batch_size=opt.train_batch_size,
                             shuffle=True, num_workers=0)
    dlen = loader.dataset.__len__()
    dbs = loader.batch_size
    batch_last_it = dlen // dbs - 1
    return loader,batch_last_it

def add_noise_to_image(clean, sigma):
    noisy = clean + sigma_255_to_torch(sigma) * th.randn_like(clean)
    return noisy

def sigma_255_to_torch(sigma_255):
    return (sigma_255 / 255) / 0.5
