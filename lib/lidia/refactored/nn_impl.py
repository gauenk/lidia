
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- diff. non-local search --
import dnls

# -- separate class and logic --
from n4net.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -- helper imports --
from n4net.utils.inds import get_3d_inds
from .misc import get_image_params

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_nn0(self,image_n,srch_img=None,flows=None,train=False,ws=29,wt=0):

    #
    # -- Our Search --
    #

    # -- pad & unpack --
    ps = self.patch_w
    neigh_pad = 14
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- prepeare image --
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)

    # -- params --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)

    # -- get search image --
    if not(srch_img is None):
        img_nn0 = self.pad_crop0(image_n, self.pad_offs, train)
        img_nn0 = self.rgb2gray(img_nn0)
    elif self.arch_opt.rgb:
        img_nn0 = self.rgb2gray(image_n0)
    else:
        img_nn0 = image_n0

    # -- get search inds --
    pad = ps//2
    t,c,h,w = image_n.shape
    hp,wp = params['patches_h'],params['patches_w']
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    # print("[stdn] image_n0.shape: ",image_n0.shape)
    # print("[stdn] image_nn0.shape: ",img_nn0.shape)
    # print("[stdn] hp,wp: ",hp,wp)

    # -- add padding --
    t,c,h0,w0 = image_n0.shape
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- search --
    k,ps,pt,chnls = 14,self.patch_w,1,1
    nlDists,nlInds = dnls.simple.search.run(img_nn0,queryInds,flows,
                                            k,ps,pt,ws,wt,chnls)

    #
    # -- Scatter Section --
    #

    # -- indexing patches --
    t,c,h,w = image_n0.shape
    patches = dnls.simple.scatter.run(image_n0,nlInds,self.patch_w)
    # print("[0] patches.shape: ",patches.shape)
    ishape = '(t p) k 1 c h w -> t p k (c h w)'
    patches = rearrange(patches,ishape,t=t)

    # -- rehape --
    patches = rearrange(patches,'t (h w) k d -> t h w k d',h=hp)
    nlInds = rearrange(nlInds,'(t h w) k tr -> t h w k tr',t=t,h=hp)
    nlDists = rearrange(nlDists,'(t h w) k -> t h w k',t=t,h=hp)

    # -- append anchor patch spatial variance --
    d = patches.shape[-1]
    patch_dist0 = nlDists[...,1:]
    patch_var0 = patches[..., [0], :].std(dim=-1).pow(2)*d
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- remove padding --
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    return patches,patch_dist0,nlInds,params

@register_method
def run_nn1(self,image_n,srch_img=None,flows=None,train=False,ws=29,wt=0):

    # -- unpack --
    t = image_n.shape[0]
    neigh_pad = 14
    ps = self.patch_w
    device = image_n.device
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]

    # -- pad & unpack --
    image_n1 = self.prepare_image_n1(image_n,train)
    params = get_image_params(image_n1, 2*self.patch_w-1, 2*neigh_pad)

    #
    #  -- DNLS Search --
    #

    # -- get search image --
    if not(srch_img is None):
        img_nn1 = self.prepare_image_n1(srch_img,train)
        img_nn1 = self.rgb2gray(img_nn1)
    elif self.arch_opt.rgb:
        img_nn1 = self.rgb2gray(image_n1)
    else:
        img_nn1 = image_n1

    # -- get search inds --
    hp,wp = params['patches_h'],params['patches_w']
    queryInds = th.arange(t*hp*wp,device=device).reshape(-1,1,1,1)
    queryInds = get_3d_inds(queryInds,hp,wp)[:,0]
    t,c,h0,w0 = image_n1.shape

    # -- inds offsets --
    sh,sw = (h0 - hp)//2,(w0 - wp)//2
    queryInds[...,1] += sh
    queryInds[...,2] += sw

    # -- exec search --
    k,pt,chnls = 14,1,1
    nlDists,nlInds = dnls.simple.search.run(img_nn1,queryInds,flows,
                                            k,ps,pt,ws,wt,chnls,
                                            stride=2,dilation=2)

    #
    # -- Scatter Section --
    #

    # -- dnls --
    patches = dnls.simple.scatter.run(image_n1,nlInds,ps,dilation=2)

    #
    # -- Final Formatting --
    #

    # - reshape --
    hp,wp = params['patches_h'],params['patches_w']
    nlDists = rearrange(nlDists,'(t h w) k -> t h w k',t=t,h=hp)
    nlInds = rearrange(nlInds,'(t h w) k tr -> t h w k tr',t=t,h=hp)
    ishape = '(t ih iw) k 1 c h w -> t ih iw k (c h w)'
    patches = rearrange(patches,ishape,ih=hp,iw=wp)

    # -- patch variance --
    d = patches.shape[-1]
    patch_var = patches[...,0,:].std(-1)**2*d
    nlDists[...,:-1] = nlDists[...,1:]
    nlDists[...,-1] = patch_var

    # -- remove padding --
    t,c,h1,w1 = image_n1.shape
    sh,sw = (h1 - hp)//2,(w1 - wp)//2
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    return patches,nlDists,nlInds,params
