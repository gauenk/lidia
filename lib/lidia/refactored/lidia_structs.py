
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- submodules --
from .pdn_structs import PatchDenoiseNet

# -- neural network --
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import fold

# -- diff. non-local search --
import dnls

# -- utils --
from lidia.utils import clean_code
from lidia.utils import gpu_mem

# -- dev basics --
from dev_basics import flow
from dev_basics.utils.timer import ExpTimerList,ExpTimer
from dev_basics.utils import clean_code

# -- separate logic --
from . import adapt
from . import adapt_og
from . import nn_impl
from . import nn_impl_orig

# -- misc imports --
from .misc import crop_offset

class BaseLIDIA(nn.Module):

    def __init__(self, pad_offs, arch_opt, gpu_stats=False):
        super(BaseLIDIA, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs
        self.gpu_stats = gpu_stats
        self.verbose = False

        self.nn_recording = True
        self.mem_res = -1
        self.mem_alloc = -1

        self.patch_w = 5 if arch_opt.rgb else 7
        self.ver_size = 80 if arch_opt.rgb else 64

        self.rgb2gray = nn.Conv2d(in_channels=3, out_channels=1,
                                  kernel_size=(1, 1), bias=False)
        self.rgb2gray.weight.data = th.tensor([0.2989, 0.5870, 0.1140],
                                                 dtype=th.float32).view(1, 3, 1, 1)
        self.rgb2gray.weight.requires_grad = False

        kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                       kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False
        self.ws = 29
        self.wt = 0
        self.div = False
        self.rescale = True
        self.train = False
        self.batch_size = None
        self.batch_alpha = None
        self.srch_img = None
        self.use_timer = False
        self.times = ExpTimerList(self.use_timer)

        self.pdn = PatchDenoiseNet(arch_opt=arch_opt,patch_w=self.patch_w,
                                   ver_size=self.ver_size,gpu_stats=self.gpu_stats)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Forward Pass
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def forward(self, noisy, flows=None):
        """

        Primary Network Backbone

        "batch_size,batch_alpha" only included to match api call

        """

        #
        # -- Prepare --
        #


        # -- unpack --
        div = self.div
        rescale = self.rescale
        train = self.train
        batch_size = self.batch_size
        batch_alpha = self.batch_alpha
        srch_img = self.srch_img
        ws,wt = self.ws,self.wt

        # -- normalize for input ---
        noisy = noisy.clone()
        if div: noisy = noisy/255.
        if rescale: noisy = (noisy - 0.5)/0.5
        means = noisy.mean((-2,-1),True)
        noisy -= means
        if srch_img is None: srch_img = noisy
        else: srch_img = srch_img/255.

        #
        # -- Non-Local Search --
        #

        # -- recording --
        if self.nn_recording:
            gpu_mem.peak_gpu_stats("pre-nn",reset=True)

        # -- [nn0 search]  --
        output0 = self.run_nn0(noisy,srch_img,flows,train,ws=ws,wt=wt)
        patches0 = output0[0]
        dists0 = output0[1]
        inds0 = output0[2]
        params0 = output0[3]

        # -- mark recoding --
        if self.nn_recording:
            mem_res,mem_alloc = gpu_mem.peak_gpu_stats("post-nn",reset=True)
            self.mem_res = mem_res
            self.mem_alloc = mem_alloc

        # -- [nn1 search]  --
        output1 = self.run_nn1(noisy,srch_img,flows,train,ws=ws,wt=wt)
        patches1 = output1[0]
        dists1 = output1[1]
        inds1 = output1[2]
        params1 = output1[3]

        #
        # -- Patch-based Denoising --
        #

        # -- reshape --
        patches0 = rearrange(patches0,'t h w k d -> t (h w) k d')
        dists0 = rearrange(dists0,'t h w k -> t (h w) k')
        inds0 = rearrange(inds0,'t h w k tr -> t (h w) k tr')
        patches1 = rearrange(patches1,'t h w k d -> t (h w) k d')
        dists1 = rearrange(dists1,'t h w k -> t (h w) k')
        inds1 = rearrange(inds1,'t h w k tr -> t (h w) k tr')

        # -- exec --
        deno,patches_w = self.pdn(patches0,dists0,inds0,params0,
                                  patches1,dists1,inds1,params1)
        #
        # -- Final Weight Aggregation --
        #

        deno = self.run_parts_final(deno,patches_w,inds0,params0)
        assert th.any(th.isnan(deno)).item() is False

        #
        # -- Format --
        #

        # -- normalize for output ---
        deno += means # normalize
        noisy += means # restore
        if rescale:
            deno[...]  = (deno  * 0.5 + 0.5) # normalize
            noisy[...] = (noisy * 0.5 + 0.5) # restore
        if div:
            deno[...]  = 255.*deno
            noisy[...] = 255.*noisy
        return deno

    def run_parts_final(self,image_dn,patch_weights,inds,params):

        # -- prepare --
        c = 3
        ps = self.patch_w
        pdim = image_dn.shape[-1]
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
        wpatches = (patch_weights * ones_tmp).transpose(2, 1)
        image_dn = image_dn.transpose(2, 1)
        # print("image_dn.shape: ",image_dn.shape)

        # -- prepare gather --
        t,hw,k,tr = inds.shape
        inds = rearrange(inds[...,0,:],'t p tr -> (t p) 1 tr').clone()

        # -- inds --
        inds[:,:,1] += (ps//2)
        inds[:,:,2] += (ps//2)

        # -- fold --
        h,w = params['pixels_h'],params['pixels_w']
        shape = (h,w)
        # print("final fold: ",h,w)
        image_dn = fold(image_dn,shape,(ps,ps))
        patch_cnt = fold(wpatches,shape,(ps,ps))

        # # -- prepare --
        # h,w = params['pixels_h'],params['pixels_w']
        # shape = (t,c,h,w)
        # zeros = th.zeros_like(inds[:,:,0],dtype=th.float32,device=inds.device)
        # image_dn = rearrange(image_dn,'t (c h w) n -> (t n) 1 1 c h w',h=ps,w=ps)
        # wpatches = rearrange(wpatches,'t (c h w) n -> (t n) 1 1 c h w',h=ps,w=ps)

        # # -- process --
        # image_dn,_ = dnls.simple.gather.run(image_dn, zeros, inds, shape=shape)
        # patch_cnt,_ = dnls.simple.gather.run(wpatches, zeros, inds, shape=shape)

        # -- crop --
        # print(params['patches_h'])
        row_offs = min(ps - 1, params['patches_h'] - 1)
        col_offs = min(ps - 1, params['patches_w'] - 1)
        # print("row_offs,col_offs: ",row_offs,col_offs)
        # print("[stnd] image_dn.shape: ",image_dn.shape)
        # image_dn = image_dn[:,:,4:-4,4:-4]
        # image_dn /= patch_cnt[:,:,4:-4,4:-4]
        image_dn = crop_offset(image_dn, (row_offs,), (col_offs,))
        image_dn /= crop_offset(patch_cnt, (row_offs,), (col_offs,))
        # print("[stnd-post] image_dn.shape: ",image_dn.shape)
        # print("[stnd] image_dn.shape: ",image_dn.shape)

        return image_dn

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #           Padding & Cropping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def pad_crop0(self, image, pad_offs, train):
        return self._pad_crop0(image, pad_offs, train, self.patch_w)

    @staticmethod
    def _pad_crop0(image,pad_offs,train,patch_w):
        if not train:
            reflect_pad = [patch_w - 1] * 4
            constant_pad = [14] * 4
            image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'),
                                constant_pad, 'constant', -1)
        else:
            image = crop_offset(image, (pad_offs,), (pad_offs,))
            # print("[pad_crop0] image.shape: ",image.shape)
        return image

    def pad_crop1(self, image, train, mode):
        return self._pad_crop1(image, train, mode, self.patch_w)

    @staticmethod
    def _pad_crop1(image, train, mode, patch_w):
        if not train:
            if mode == 'reflect':
                bilinear_pad = 1
                averaging_pad = (patch_w - 1) // 2
                patch_w_scale_1 = 2 * patch_w - 1
                find_nn_pad = (patch_w_scale_1 - 1) // 2
                reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
                image = nn_func.pad(image, reflect_pad, 'reflect')
            elif mode == 'constant':
                constant_pad = [28] * 4
                image = nn_func.pad(image, constant_pad, 'constant', -1)
            else:
                assert False
        return image

    def prepare_image_n1(self,image_n,train):

        # -- pad & unpack --
        patch_numel = (self.patch_w ** 2) * image_n.shape[1]
        device = image_n.device
        image_n1 = self.pad_crop1(image_n, train, 'reflect')
        im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape

        # -- bilinear conv & crop --
        image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
        image_n1 = self.bilinear_conv(image_n1)
        image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)
        image_n1 = self.pad_crop1(image_n1, train, 'constant')
        return image_n1

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Nearest Neighbor Searches
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def run_nn0(self):
        raise NotImplemented("")

    def run_nn1(self):
        raise NotImplemented("")


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#        Derivative Classes
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@clean_code.add_methods_from(adapt)
@clean_code.add_methods_from(nn_impl)
class LIDIA(BaseLIDIA):
    def __init__(self, pad_offs, arch_opt, gpu_stats=False):
        super(LIDIA, self).__init__(pad_offs, arch_opt, gpu_stats)

# @clean_code.add_methods_from(adapt_og)
@clean_code.add_methods_from(adapt)
@clean_code.add_methods_from(nn_impl_orig)
class OriginalLIDIA(BaseLIDIA):
    def __init__(self, pad_offs, arch_opt, gpu_stats=False):
        super(OriginalLIDIA, self).__init__(pad_offs, arch_opt, gpu_stats)

class ArchitectureOptions:
    def __init__(self, rgb):
        self.rgb = rgb

