"""

Lidia's Non-Local Search

"""


# -- nn --
import torch as th
from torch.nn.functional import conv2d
import torch.nn.functional as nn_func

# -- lidia imports --
from ..batched.misc import crop_offset,get_npatches
# from ..batched import nn_impl

# -- separate class and logic --
# from lidia.utils import clean_code
# __methods__ = []
# register_method = clean_code.register_method(__methods__)
# @clean_code.add_methods_from(nn_impl)

class LNLSearch():

    def __init__(self,k,ps,ws,stride0,train=False,lidia_pad=True):

        # -- extract --
        self.k = k
        self.ps = ps
        self.ws = ws
        self.wt = 0
        self.stride0 = stride0
        self.train = train
        self.lidia_pad = lidia_pad
        self.pad_offs = calc_padding(self.ws,self.k-1)


        # -- rgb to gray --
        self.rgb2gray = th.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 1), bias=False)
        data = th.tensor([0.2989, 0.5870, 0.1140],dtype=th.float32).view(1, 3, 1, 1)
        self.rgb2gray.weight.data = data
        self.rgb2gray.weight.requires_grad = False

        # -- check --
        neigh_patches_w = 2 * (self.k-1) + 1
        assert self.ws == neigh_patches_w
        assert self.stride0 == 1,"Must be 1"

    def __call__(self,vid):

        # -- unpack --
        H,W = vid.shape[-2:]
        B = vid.shape[0]
        self.rgb2gray = self.rgb2gray.to(vid.device)

        # -- prepare video --
        dists,inds = [],[]
        for b in range(B):

            # -- run search --
            vid_pad = self.pad_crop0(vid[b],self.pad_offs,self.train)
            vid_pad = self.rgb2gray(vid_pad)
            dists_b,inds_b = self.find_nn(vid_pad,H,W,scale=0,case=None)

            # -- agg --
            dists.append(dists_b)
            inds.append(inds_b)

        dists = th.stack(dists)
        inds = th.stack(inds)
        return dists,inds

    def find_nn(self, vid_pad, H, W, scale=0, case=None):

        # -- unpack --
        B = vid_pad.shape[0]
        NH = self.k-1

        # -- num patches --
        nH = (H-(self.ps-1)-1)//self.stride0+1
        nW = (W-(self.ps-1)-1)//self.stride0+1

        # -- alloc --
        top_dist = th.zeros(B, nH, nW, self.ws ** 2,
                            device=vid_pad.device).fill_(float('nan'))
        dfilter = th.ones(1, 1, self.ps, self.ps, device=vid_pad.device)
        y = vid_pad[:, :, NH:NH + H, NH:NH + W]

        # -- differences --
        for row in range(self.ws):
            for col in range(self.ws):
                lin_ind = row * self.ws + col
                y_shift = vid_pad[:, :,row:row + H, col:col + W]
                top_dist[:, :, :, lin_ind] = conv2d((y_shift - y) ** 2, dfilter).squeeze(1)

        # -- topk --
        top_dist, top_ind = th.topk(top_dist, NH, dim=3, largest=False, sorted=True)

        # -- unravel top_ind --
        top_ind_rows = th.div(top_ind,self.ws,rounding_mode="trunc")
        top_ind_cols = top_ind % self.ws
        col_arange = th.arange(nW,device=vid_pad.device).view(1, 1, -1, 1)
        row_arange = th.arange(nH,device=vid_pad.device).view(1, -1, 1, 1)
        if scale == 1:
            if case == '00':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '10':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + col_arange * 2
            elif case == '01':
                top_ind_rows = top_ind_rows * 2 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            elif case == '11':
                top_ind_rows = top_ind_rows * 2 + 1 + row_arange * 2
                top_ind_cols = top_ind_cols * 2 + 1 + col_arange * 2
            else:
                assert False
        elif scale == 0:
            top_ind_rows += row_arange
            top_ind_cols += col_arange
        else:
            assert False
        nWp = vid_pad.shape[-2] - (self.ps - 1)
        top_ind = top_ind_rows * nWp + top_ind_cols

        return top_dist, top_ind

    def pad_crop0(self, image, pad_offs, train):
        if not train:
            reflect_pad = [self.ps-1] * 4
            constant_pad = [self.k-1] * 4
            image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'),
                                constant_pad, 'constant', -1)
        else:
            image = crop_offset(image, (pad_offs,), (pad_offs,))

        return image

    def flops(self,B,C,H,W):
        Cout = 1
        P = self.ps
        W = self.ws
        nH = (H-1)//self.stride0+1
        nW = (W-1)//self.stride0+1
        nflops_rc = 2 * P * P * C * nH * nW * Cout
        nflops = nflops_rc * (W**2)
        return B * nflops

    def flops_bwd(self,B,C,H,W):
        pass

    def radius(self,*args,**kwargs):
        return self.ws

def calc_padding(patch_w=5,k=14):
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + (k-1)
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + k * 2
    offs = total_pad - total_pad0
    return offs,total_pad
