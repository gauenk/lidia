
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- management --
from easydict import EasyDict as edict

# -- submodules --
from .pdn_structs import PatchDenoiseNet

# -- neural network --
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.nn.functional import fold
from torch.nn.functional import pad as nn_pad
from torchvision.transforms.functional import center_crop

# -- diff. non-local search --
import dnls

# -- separate logic --
from . import adapt
from . import adapt_og
from . import nn_impl
from . import im_shapes

# -- utils --
from lidia.utils import clean_code

# -- misc imports --
from .misc import calc_padding
from .misc import crop_offset,get_npatches,get_step_fxns,assert_nonan
from lidia.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

@clean_code.add_methods_from(adapt)
@clean_code.add_methods_from(adapt_og)
@clean_code.add_methods_from(im_shapes)
@clean_code.add_methods_from(nn_impl)
class BatchedLIDIA(nn.Module):

    def __init__(self, pad_offs, arch_opt, lidia_pad=False,
                 match_bn=False,remove_bn=False,grad_sep_part1=True,
                 name="",verbose=False):
        super(BatchedLIDIA, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs

        # -- modify changes --
        self.lidia_pad = lidia_pad
        self.match_bn = match_bn
        self.remove_bn = remove_bn
        self.grad_sep_part1 = grad_sep_part1
        self.gpu_stats = False
        self.name = name
        self.verbose = verbose

        self.patch_w = 5 if arch_opt.rgb else 7
        self.ps = self.patch_w
        self.k = 14
        self.neigh_pad = self.k
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

        self.pdn = PatchDenoiseNet(arch_opt=arch_opt,patch_w=self.patch_w,
                                   ver_size=self.ver_size,
                                   gpu_stats=self.gpu_stats,
                                   match_bn=self.match_bn,
                                   remove_bn=self.remove_bn,
                                   grad_sep_part1=self.grad_sep_part1,
                                   name=name)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #          Forward Pass
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def forward(self, noisy, sigma, srch_img=None, flows=None,
                ws=29, wt=0, train=False, rescale=True, stride=1,
                batch_size = -1, batch_alpha = 0.5, region=None):
        """

        Primary Network Backbone

        """

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # --     Prepare       --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- normalize for input ---
        self.print_gpu_stats("Init")
        if rescale: noisy = (noisy/255. - 0.5)/0.5
        means = noisy.mean((-2,-1),True)
        noisy -= means
        if srch_img is None:
            srch_img = noisy
        noisy = noisy.contiguous()

        # -- unpack --
        device = noisy.device
        vshape = noisy.shape
        t,c,h,w = noisy.shape
        ps,pt = self.patch_w,1

        # -- assign for match_bn check --
        self.pdn.nframes = noisy.shape[0]
        self.pdn.sep_net.nframes = noisy.shape[0]

        # -- get num of patches --
        hp,wp = get_npatches(vshape, train, self.ps, self.pad_offs,
                             self.k, self.lidia_pad)

        # -- get inset region --
        noregion = region is None
        if region is None:
            region = [0,t,0,0,hp,wp]
        else:
            assert self.lidia_pad is False,"Can't do a region and match lidia."
            assert region[-1] <= w,"Must be within vid frame size."
            assert region[-2] <= h,"Must be within vid frame size."
            region = [r for r in region] # copy
            if len(region) == 4: # spatial onyl; add time
                region = [0,t,] + region
            pad = 2*(self.ps//2)
            region[4] += pad
            region[5] += pad
            hp = region[4] - region[2]
            wp = region[5] - region[3]
        t = region[1] - region[0] # frames to deno

        # -- patch-based functions --
        levels = self.get_levels()
        pfxns = edict()
        for lname,params in levels.items():
            dil = params['dil']
            h_l,w_l,pad_l = self.image_shape((hp,wp),ps,dilation=dil)
            region_l = [pad_l,pad_l,hp+pad_l,wp+pad_l]
            vshape_l = (t,c,h_l,w_l)
            pfxns[lname] = get_step_fxns(vshape_l,region_l,ps,stride,dil,device)

        # -- allocate final video  --
        deno_folds = self.allocate_final(t,c,hp,wp)
        self.print_gpu_stats("Alloc")


        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # --    First Step     --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- Batching Info --
        if self.verbose: print("batch_size: ",batch_size)
        nqueries = t * ((hp-1)//stride+1) * ((wp-1)//stride+1)
        if batch_size <= 0: batch_size_step = nqueries
        else: batch_size_step = batch_size
        nbatches = (nqueries - 1)//batch_size_step+1

        for batch in range(nbatches):

            # -- Info --
            if self.verbose:
                print("[Step0] Batch %d/%d" % (batch+1,nbatches))

            # -- Batching Inds --
            qindex = min(batch * batch_size_step,nqueries)
            batch_size_i = min(batch_size_step,nqueries - qindex)
            queries = dnls.utils.inds.get_iquery_batch(qindex,batch_size_i,
                                                       stride,region,t,device)
            th.cuda.synchronize()

            # -- Process Each Level --
            for level in levels:
                pfxns_l,params_l = pfxns[level],levels[level]

                # -- Non-Local Search --
                nn_info = params_l.nn_fxn(noisy,queries,pfxns_l.scatter,
                                          srch_img,flows,train,ws=ws,wt=wt)

                # -- Patch-based Denoising --
                self.pdn.batched_step(nn_info,pfxns_l,params_l,level,qindex)

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # -- Normalize Videos --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        for level in levels:
            vid = pfxns[level].fold.vid
            wvid = pfxns[level].wfold.vid
            vid_z = vid / wvid
            assert_nonan(vid_z)
            levels[level]['vid'] = vid_z
            del wvid

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # --    Second Step    --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- shrink batch size since all patches at once --
        if batch_size > 0: batch_size = int(batch_alpha * batch_size)
        else: batch_size_step = nqueries
        nbatches = (nqueries - 1)//batch_size_step+1

        # -- for each batch --
        for batch in range(nbatches):

            # -- Info --
            if self.verbose:
                print("[Step1] Batch %d/%d" % (batch+1,nbatches))

            #
            # -- Batching Inds --
            #

            qindex = min(batch * batch_size_step,nqueries)
            batch_size_i = min(batch_size_step,nqueries - qindex)
            queries = dnls.utils.inds.get_iquery_batch(qindex,batch_size_i,
                                                       stride,region,t,device)


            #
            # -- Non-Local Search @ Each Level --
            #

            nn_info = {}
            for level in levels:
                nn_fxn = levels[level]['nn_fxn']
                scatter = pfxns[level].scatter
                nn_info_l = nn_fxn(noisy,queries,scatter,srch_img,
                                   flows,train,ws=ws,wt=wt)
                nn_info[level] = nn_info_l

            #
            # -- Patch Denoising --
            #

            pdeno,wpatches = self.pdn.batched_fwd_b(levels,nn_info,pfxns,
                                                    qindex,batch_size_i)
            assert_nonan(pdeno)
            assert_nonan(wpatches)

            #
            # -- Final Weight Aggregation --
            #

            self.run_parts_final(pdeno,wpatches,qindex,
                                 deno_folds.img,deno_folds.wimg)

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # --    Final Format    --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- Unpack --
        deno = self.final_format(deno_folds.img,deno_folds.wimg)
        assert_nonan(deno)

        # -- Normalize for output ---
        deno += means[region[0]:region[1]] # normalize
        noisy += means # restore
        if rescale:
            deno[...]  = 255.*(deno  * 0.5 + 0.5) # normalize
            noisy[...] = 255.*(noisy * 0.5 + 0.5) # restore
        return deno

    def allocate_final(self,t,c,hp,wp):
        coords = [0,0,hp,wp]
        folds = edict()
        folds.img = dnls.ifold.iFold((t,c,hp,wp),coords,stride=1,dilation=1)
        folds.wimg = dnls.ifold.iFold((t,c,hp,wp),coords,stride=1,dilation=1)
        return folds

    def run_parts_final(self,image_dn,patch_weights,qindex,fold_nl,wfold_nl):

        # -- expands wpatches --
        pdim = image_dn.shape[-1]
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
        wpatches = (patch_weights * ones_tmp)

        # -- format to fold --
        ps = self.patch_w
        shape_str = 't n (c h w) -> (t n) 1 1 c h w'
        image_dn = rearrange(image_dn,shape_str,h=ps,w=ps)
        wpatches = rearrange(wpatches,shape_str,h=ps,w=ps)

        # -- contiguous --
        image_dn = image_dn.contiguous()
        wpatches = wpatches.contiguous()

        # -- dnls fold --
        image_dn = fold_nl(image_dn,qindex)
        patch_cnt = wfold_nl(wpatches,qindex)

    def final_format(self,fold_nl,wfold_nl):
        # -- crop --
        pad = self.ps//2
        image_dn = fold_nl.vid
        patch_cnt = wfold_nl.vid
        image_dn = image_dn[:,:,pad:-pad,pad:-pad]
        image_dn /= (patch_cnt[:,:,pad:-pad,pad:-pad] + 1e-10)
        return image_dn

    def get_levels(self):
        levels = {"l0":{"dil":1,
                        "wdiv":False,
                        "nn_fxn":self.run_nn0},
                  "l1":{"dil":2,
                        "wdiv":True,
                        "nn_fxn":self.run_nn1},
        }
        levels = edict(levels)
        return levels

    def print_gpu_stats(self,name="-"):
        print_gpu_stats(self.gpu_stats,name)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #         Padding & Cropping
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- pad/crops --
    def pad_crop0(self):
        raise NotImplemented("")

    def _pad_crop0_eff():
        raise NotImplemented("")

    def _pad_crop0_og():
        raise NotImplemented("")

    def pad_crop1(self):
        raise NotImplemented("")

    def _pad_crop1(self):
        raise NotImplemented("")

    # -- shapes --
    def image_n0_shape(self):
        raise NotImplemented("")

    def image_n0_shape_og(self):
        raise NotImplemented("")

    def image_n1_shape(self):
        raise NotImplemented("")

    def image_n1_shape_og(self):
        raise NotImplemented("")

    # -- prepare --
    def prepare_image_n1(self):
        raise NotImplemented("")

    def prepare_image_n1_eff(self):
        raise NotImplemented("")

    def prepare_image_n1_og(self):
        raise NotImplemented("")

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #      Nearest Neighbor Searches
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def run_nn0(self):
        raise NotImplemented("")

    def run_nn1(self):
        raise NotImplemented("")

class ArchitectureOptions:
    def __init__(self, rgb):
        self.rgb = rgb

