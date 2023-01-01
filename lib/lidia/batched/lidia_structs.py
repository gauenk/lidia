
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
from . import nn_timer

# -- utils --
from lidia.utils import clean_code
from lidia.utils import gpu_mem

# -- misc imports --
from .misc import calc_padding
from .misc import crop_offset,get_npatches,get_step_fxns,assert_nonan
from lidia.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats

# -- dev basics --
from dev_basics import flow
from dev_basics.utils.timer import ExpTimerList,ExpTimer
from dev_basics.utils import clean_code

@clean_code.add_methods_from(nn_timer)
@clean_code.add_methods_from(adapt)
@clean_code.add_methods_from(adapt_og)
@clean_code.add_methods_from(im_shapes)
@clean_code.add_methods_from(nn_impl)
class BatchedLIDIA(nn.Module):

    def __init__(self, adapt_cfg, pad_offs, arch_opt, lidia_pad=False,
                 match_bn=False,remove_bn=False,grad_sep_part1=True,
                 name="",ps=5,ws=29,wt=0,stride=1,bs=-1,bs_te=-1,
                 bs_alpha=0.25, idiv=False, rescale=True,
                 nn_recording=False,nn_record_first_only = False,
                 use_nn_timer=False, verbose=False):
        super(BatchedLIDIA, self).__init__()
        self.arch_opt = arch_opt
        self.pad_offs = pad_offs

        # -- record nn gpu mem --
        self.use_timer = use_nn_timer
        self.times = ExpTimerList(self.use_timer)
        self.nn_recording = nn_recording
        self.nn_record_first_only = nn_record_first_only
        self.mem_res = -1
        self.mem_alloc = -1
        self.imax = 1.

        # -- modify changes --
        self.lidia_pad = lidia_pad
        self.match_bn = match_bn
        self.remove_bn = remove_bn
        self.grad_sep_part1 = grad_sep_part1
        self.gpu_stats = False
        self.name = name
        self.verbose = verbose

        self.ws = ws
        self.wt = wt
        self.stride = stride
        self.bs = bs
        self.bs_te = bs_te
        self.bs_alpha = bs_alpha
        self.adapt_cfg = adapt_cfg
        self.idiv = idiv
        self.rescale = rescale

        # self.patch_w = 5 if arch_opt.rgb else 7
        self.ps = ps#self.patch_w
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

        self.pdn = PatchDenoiseNet(arch_opt=arch_opt,patch_w=self.ps,
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

    def forward(self, noisy, srch_img=None, flows=None, region=None):

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
        if self.idiv: noisy = noisy/self.imax#255.
        if self.rescale: noisy = (noisy - 0.5)/0.5
        # if rescale: noisy = (noisy/self.imax - 0.5)/0.5
        means = noisy.mean((-2,-1),True)
        noisy -= means
        if srch_img is None:
            srch_img = noisy
        noisy = noisy.contiguous()

        # -- unpack --
        device = noisy.device
        vshape = noisy.shape
        t,c,h,w = noisy.shape
        ws,wt = self.ws,self.wt
        ps,pt = self.ps,1
        stride = self.stride
        batch_size = self.bs_te if self.training else self.bs
        batch_alpha = self.bs_alpha
        train = self.training

        # -- no batch flows --
        flows = flow.remove_batch(flows)

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
            vshape_l = (1,t,c,h_l,w_l)
            pfxns[lname] = get_step_fxns(vshape_l,region_l,ps,stride,dil,device)

        # -- allocate final video  --
        deno_folds = self.allocate_final(t,c,hp,wp)
        self.print_gpu_stats("Alloc")


        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # --    First Step     --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-


        # -- timer --
        timer_attn = ExpTimer(self.use_timer)
        timer_attn.sync_start("attn")
        rec_mem = True # for gpu recording

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
            # print("batch_size_i: ",batch_size_i)
            queries = dnls.utils.inds.get_iquery_batch(qindex,batch_size_i,
                                                       stride,region,t,device)
            th.cuda.synchronize()
            # print("qindex: ",qindex,batch_size_i,nqueries)

            # -- Process Each Level --
            for level in levels:

                # -- timer --
                timer = ExpTimer(self.use_timer)

                # -- unpack --
                pfxns_l,params_l = pfxns[level],levels[level]

                # -- recording nn gpu mem --
                if self.nn_recording and level == "l0":
                    gpu_mem.peak_gpu_stats("pre-nn",reset=True)
                timer.sync_start("search")


                # -- Non-Local Search --

                nn_info = params_l.nn_fxn(noisy,queries,pfxns_l.scatter,
                                          srch_img,flows,train,ws=ws,wt=wt)

                # -- recording nn gpu mem --
                timer.sync_stop("search")
                if self.nn_recording and level == "l0":
                    mem_res,mem_alloc = gpu_mem.peak_gpu_stats("post-nn",reset=True)
                    if rec_mem:
                        self.mem_res = mem_res
                        self.mem_alloc = mem_alloc
                        if self.nn_record_first_only: rec_mem = False
                timer.sync_start("agg")

                # -- Patch-based Denoising --
                self.pdn.batched_step(nn_info,pfxns_l,params_l,level,qindex)

                # -- timer --
                timer.sync_stop("agg")
                self._update_times(timer)

        # -=-=-=-=-=-=-=-=-=-=-=-
        #
        # -- Normalize Videos --
        #
        # -=-=-=-=-=-=-=-=-=-=-=-

        # -- timing update --
        timer_attn.sync_stop("attn")
        self._update_times(timer_attn)

        # -- normalize --
        for level in levels:
            vid = pfxns[level].fold.vid[0]
            wvid = pfxns[level].wfold.vid[0]
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
        if batch_size > 0: batch_size_step = int(batch_alpha * batch_size)
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
        if self.rescale:
            deno[...]  = deno  * 0.5 + 0.5 # normalize
            noisy[...] = noisy * 0.5 + 0.5 # restore
        if self.idiv:
            deno[...] = 255.*deno
            noisy[...] = 255.*noisy
        return deno

    def restore_vid(self,vid,means,rescale):
        vid += means # restore
        if rescale:
            vid[...] = self.imax*(vid * 0.5 + 0.5) # restore
        return vid

    def allocate_final(self,t,c,hp,wp):
        coords = [0,0,hp,wp]
        folds = edict()
        folds.img = dnls.iFold((1,t,c,hp,wp),coords,stride=1,dilation=1)
        folds.wimg = dnls.iFold((1,t,c,hp,wp),coords,stride=1,dilation=1)
        return folds

    def run_parts_final(self,image_dn,patch_weights,qindex,fold_nl,wfold_nl):

        # -- expands wpatches --
        pdim = image_dn.shape[-1]
        image_dn = image_dn * patch_weights
        ones_tmp = th.ones(1, 1, pdim, device=image_dn.device)
        wpatches = (patch_weights * ones_tmp)

        # -- format to fold --
        ps = self.ps
        shape_str = 't n (c h w) -> (t n) 1 1 c h w'
        image_dn = rearrange(image_dn,shape_str,h=ps,w=ps)
        wpatches = rearrange(wpatches,shape_str,h=ps,w=ps)

        # -- contiguous --
        image_dn = image_dn.contiguous()
        wpatches = wpatches.contiguous()

        # -- dnls fold --
        image_dn = fold_nl(image_dn[None,:],qindex)[0]
        patch_cnt = wfold_nl(wpatches[None,:],qindex)[0]

    def final_format(self,fold_nl,wfold_nl):
        # -- crop --
        pad = self.ps//2
        image_dn = fold_nl.vid[0]
        patch_cnt = wfold_nl.vid[0]
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
    #             Timing
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # def update_times(self):
    #     for i in range(self.nblocks-1):
    #         self._update_times(self.nls[i].times)
    #         self.nls[i]._reset_times()

    def reset_times(self):
        # for i in range(self.nblocks-1):
        #     self.nls[i]._reset_times()
        self._reset_times()

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

