# -- rand nums --
import random

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- image pair dataset --
import random
from PIL import Image
import torch.utils.data as data
import torch.nn.functional as nnf
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

def create_sobel_filter():
    # -- get sobel filter to detect rough spots --
    sobel = th.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,3,3)
    sobel_t = sobel_t.reshape(1,3,3)
    weights = th.stack([sobel,sobel_t],dim=0)
    return weights

def apply_sobel_filter(image):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    C = image.shape[-3]
    weights = create_sobel_filter()
    weights = weights.to(image.device)
    weights = repeat(weights,'b 1 h w -> b c h w',c=C)
    edges = nnf.conv2d(image,weights,padding=1,stride=1)
    edges = ( edges[:,0]**2 + edges[:,1]**2 ) ** (0.5)

    # -- compute spatial average (to find "good points") --
    weights = th.FloatTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.
    weights = weights[None,None,:,:].to(image.device)
    spatial_ave_edges = nnf.conv2d(edges[:,None,:,:],weights,padding=1,stride=1)
    return spatial_ave_edges

def point2range(p,reg,lb,ub):

    # -- pmin --
    pmin = p-((reg-1)//2+1)
    pmin = max(pmin,0)

    # -- p max --
    pmax = pmin + reg
    pmax = min(pmax,ub)

    # -- shift left if needed --
    curr_reg = pmax - pmin # current size
    lshift = reg - curr_reg # remaining
    pmin -= lshift

    # -- assign --
    pstart = pmin
    pend = pmax

    # -- assert --
    assert (pstart >= lb) and (pend <= ub)
    assert (pend - pstart) == reg
    return pstart,pend

def rslice(vid,region):
    t0,t1,h0,w0,h1,w1 = region
    return region[t0:t1,:,h0:h1,w0:w1]

class RegionProposalData():

    def __init__(self, clean, mtype, region_template, nlevels=3):

        # -- init --
        self.clean = clean
        self.mtype = mtype
        self.nlevels = nlevels

        # -- init shape --
        self.nframes = clean.shape[0]
        self.height = clean.shape[2]
        self.width = clean.shape[3]

        # -- init region --
        rtemp = region_template.split("_")
        self.reg_nframes = int(rtemp[0])
        self.reg_height = int(rtemp[1])
        self.reg_width = int(rtemp[2])

        # -- sample region via multi-scale sobels --
        self.sobels = []
        for level in range(self.nlevels):
            if level > 0:
                h,w = self.sobels[-1].shape[-2:]
                rh,rw = int(0.5*h),int(0.5*w)
                img = tf.resize(clean,(rh,rw))
            else: img = clean
            sobel = apply_sobel_filter(img)
            self.sobels.append(sobel)

    def sample_point(self):
        if self.mtype == "rand" or self.mtype == "default":
            return self.sample_random_point()
        elif self.mtype == "sobel":
            return self.sample_sobel_point()
        else:
            raise NotImplementedError(f"Uknown sample method [{self.mtype}]")

    def sample_random_point(self):
        t,c,h,w = self.clean.shape

        # -- sample t --
        ones = th.ones(t)
        rt = int(th.multinomial(ones,1).item())

        # -- sample h --
        ones = th.ones(h)
        rh = int(th.multinomial(ones,1).item())
        rh = int(th.multinomial(ones,1).item())

        # -- sample w --
        ones = th.ones(w)
        rw = int(th.multinomial(ones,1).item())

        # -- agg point --
        point = [rt,rh,rw]
        return point

    def sample_sobel_point(self):
        # print("self.sobel.shape: ",self.sobels[-1].shape)
        t,c,h,w = self.sobels[-1].shape
        size = th.Tensor([t*h*w])
        hw = h * w
        ind = int(th.multinomial(size,1).item())
        ti = ind // hw
        hi = (ind%hw)//h
        wi = (ind%hw)%w
        point = [ti,hi,wi]
        return point
        # t0,t1 = 0,t
        # h0,h1,w0,w1 = 0,h,0,w
        # region = [t0,t1,h0,w0,h1,w1]
        # for level in reversed(range(len(self.sobels))):

        #     # -- get region to sample --
        #     sobel = self.sobels[level]
        #     sobel = rslice(sobel,region)

        #     # -- sample a point --
        #     fsobel = sobel.ravel()
        #     ind = th.multinomial(fsobel,1).item()

        #     # -- convert to absolute point value --
        #     t0,t1,h0,w0,h1,w1 = region
        #     t,c,h,w = sobel[level].shape
        #     hw = h*w
        #     ti = ind//hw + t0
        #     hi = (ind%hw)//h + h0
        #     wi = (ind%hw)%w + w0

        #     # --

        #     print(ti,hi,wi)

    def __iter__(self):
        return self

    def __next__(self):
        # -- get point in image --
        point = self.sample_point()

        # -- center region --
        fstart,fend = point2range(point[0],self.reg_nframes,0,self.nframes)
        top,btm = point2range(point[1],self.reg_height,0,self.height)
        left,right = point2range(point[2],self.reg_width,0,self.width)

        # -- create region --
        region = [fstart,fend,top,left,btm,right]

        return region

    def __len__(self):
        return 10**5#self.len



