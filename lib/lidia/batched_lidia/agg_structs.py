

# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- neural network --
import torch.nn as nn
import torch.nn.functional as nn_func

# -- differentiable non-local search --
import dnls

class Aggregation0(nn.Module):
    def __init__(self, patch_w):
        super(Aggregation0, self).__init__()
        self.patch_w = patch_w

    def batched_fwd_a(self, patches, inds, fold_nl, wfold_nl):

        # -- prepare x --
        pt,ps,t = 1,self.patch_w,patches.shape[0]
        patches = rearrange(patches,'1 n 1 (c h w) -> n 1 1 c h w',h=ps,w=ps)

        # -- exec fold --
        ones = th.ones_like(patches)
        vid = fold_nl(patches,inds) # inds == qstart
        wvid = wfold_nl(ones,inds)

        return vid

    def batched_fwd_b(self, vid, qindex, bsize, unfold):
        y_out = unfold(vid,qindex,bsize)
        y_out = rearrange(y_out,'n 1 pt c h w -> 1 n 1 (pt c h w)')
        return y_out

    def forward(self):
        raise NotImplemented("")


class Aggregation1(nn.Module):

    def __init__(self, patch_w):
        super(Aggregation1, self).__init__()
        self.patch_w = patch_w

        kernel_1d = th.tensor((1 / 4, 1 / 2, 1 / 4), dtype=th.float32)
        kernel_2d = (kernel_1d.view(-1, 1) * kernel_1d).view(1, 1, 3, 3)
        self.bilinear_conv = nn.Conv2d(in_channels=1, out_channels=1,
                                       kernel_size=(3, 3), bias=False)
        self.bilinear_conv.weight.data = kernel_2d
        self.bilinear_conv.weight.requires_grad = False

    def batched_fwd_a(self, patches, inds, fold_nl, wfold_nl):
        # tag-agg1

        # -- shapes --
        pt,ps,t = 1,self.patch_w,patches.shape[0]

        # -- prepare patches --
        patches = rearrange(patches,'1 n 1 (c h w) -> n 1 1 c h w',h=ps,w=ps)

        # -- fold --
        ones = th.ones_like(patches)
        vid = fold_nl(patches,inds) # inds == qstart
        wvid = wfold_nl(ones,inds)

        return vid

    def batched_fwd_b(self, vid, qindex, bsize, unfold):

        # -- filter --
        t,c,h,w = vid.shape
        vid = nn_func.pad(vid, [1] * 4, 'reflect').view(t*c,1,h+2,w+2)
        vid = self.bilinear_conv(vid).view(t,c,h,w)
        y_out = unfold(vid,qindex,bsize)
        y_out = rearrange(y_out,'n 1 pt c h w -> 1 n 1 (pt c h w)')
        return y_out

    def forward(self):
        raise NotImplemented("")
