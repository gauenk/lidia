
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

    def forward(self, x, nlDists, nlInds, pixels_h, pixels_w, both=False):
        # tag-agg0

        # -- prepare x --
        pt,ps,t = 1,self.patch_w,x.shape[0]
        images, patches, hor_f, ver_f = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(images * hor_f, ver_f, patches)
        x = rearrange(x,'t (c h w) p -> (t p) 1 1 c h w',c=3,h=ps,w=ps)
        _,_,pt,_,ps,ps = x.shape

        # -- [gather] non-local params --
        pad = ps//2
        _nlDists = rearrange(nlDists[:,:,0],'t p -> (t p) 1').clone()
        _nlInds = rearrange(nlInds[:,:,0],'t p thr -> (t p) 1 thr').clone()
        ones = th.zeros_like(_nlDists)
        _nlInds[...,1] += pad#(ps-1) - ps//2 # delta pads from 72 -> 68
        _nlInds[...,2] += pad#(ps-1) - ps//2

        # -- [gather] prepare out size --
        hp = pixels_h# + 2*(ps-1)
        wp = pixels_w# + 2*(ps-1)
        shape = (t,3,hp,wp)

        # -- exec scatter --
        x,wx = dnls.simple.gather.run(x,ones,_nlInds,shape=shape)

        # -- post process --
        x = x / wx
        xg = x
        # dnls.testing.data.save_burst(xg,"./output/tests/","gt_agg0")
        # print(x[0,0,:5,:5])

        # -- scatter --
        x = dnls.simple.scatter.run(x,_nlInds,ps,pt,dilation=1)
        x = rearrange(x,'(t p) 1 pt c h w -> t p 1 (pt c h w)',t=t)
        if both: return x,xg
        else: return x


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

    def forward(self, x, nlDists, nlInds, pixels_h, pixels_w, both=False):
        # tag-agg1

        # -- shapes --
        pt,ps,t = 1,self.patch_w,x.shape[0]

        # -- unpack images --
        images, patches, hor_f, ver_f = x.shape
        # x = x.permute(0, 2, 3, 1).view(images * hor_f, ver_f, patches)
        shape = (x.shape[0],3,pixels_h,pixels_w)
        x = rearrange(x,'t p 1 (c h w) -> (t p) 1 1 c h w',h=ps,w=ps)
        _,_,pt,_,ps,ps = x.shape
        _nlDists = rearrange(nlDists[:,:,0],'t p -> (t p) 1').clone()
        _nlInds = rearrange(nlInds[:,:,0],'t p thr -> (t p) 1 thr').clone()

        # -- update inds --
        pad = 2*(ps//2) # dilation "= 2"
        _nlInds[...,1] += pad
        _nlInds[...,2] += pad

        # -- gather --
        shape = (t,3,pixels_h,pixels_w)
        zeros = th.zeros_like(_nlDists)
        x,wx = dnls.simple.gather.run(x,zeros,_nlInds,shape=shape,dilation=2)
        x = x / wx
        xg = x

        # -- filter --
        t,c,h,w = x.shape
        x = nn_func.pad(x, [1] * 4, 'reflect').view(t*c,1,h+2,w+2)
        x = self.bilinear_conv(x).view(t,c,h,w)

        # -- scatter --
        x = dnls.simple.scatter.run(x,_nlInds,ps,pt,dilation=2)
        x = rearrange(x,'(t p) 1 pt c h w -> t p 1 (pt c h w)',t=t)

        if both: return x,xg
        else: return x
