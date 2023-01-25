
# -- linalg --
import torch as th
from einops import rearrange,repeat
from torch.nn.functional import pad as nn_pad

# -- data mngmnt --
from easydict import EasyDict as edict

# -- diff. non-local search --
import dnls

# -- separate class and logic --
from lidia.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -- helper imports --
from lidia.utils.inds import get_3d_inds
from .misc import get_image_params,get_npatches

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def run_nn0(self,image_n,queryInds,scatter_nl,
            srch_img=None,flows=None,train=False,ws=29,wt=0,
            neigh_pad = 14,bsize=-1):

    #
    # -- Our Search --
    #

    # -- pad & unpack --
    ps = self.ps#patch_w
    device = image_n.device
    patch_numel = (ps ** 2) * image_n.shape[1]

    # -- number of patches along (height,width) --
    t,c,h,w = image_n.shape
    vshape = image_n.shape
    hp,wp = get_npatches(vshape, train, self.ps, self.pad_offs, self.k, self.lidia_pad)

    # -- prepeare image --
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)
    _,_,ha,wa = image_n0.shape
    sh,sw = (ha-hp)//2,(wa-wp)//2
    params = None

    # -- print for testing --
    self.print_gpu_stats("NN0")

    # -- get search image --
    if not(srch_img is None):
        img_nn0 = self.pad_crop0(image_n, self.pad_offs, train)
        img_nn0 = self.rgb2gray(img_nn0[:,:3])
    elif self.arch_opt.rgb:
        img_nn0 = self.rgb2gray(image_n0[:,:3])
    else:
        img_nn0 = image_n0
    img_nn0 = img_nn0.detach()

    # -- search --
    k,ps,pt,chnls = 14,self.ps,1,1
    if self.lidia_pad:

        # -- add padding --
        queryInds[...,1] += sh
        queryInds[...,2] += sw

        nlDists,nlInds = dnls.simple.search.run(img_nn0,queryInds,flows,
                                                k,ps,pt,ws,wt,chnls,
                                                reflect_bounds=False)
        nlInds = nlInds[None,:]

        # -- remove padding --
        queryInds[...,1] -= sh
        queryInds[...,2] -= sw

    else:
        qindex = queryInds
        self.update_search_flows(self.search0,img_nn0[None,:].shape,
                                 img_nn0.device,flows)
        nlDists,nlInds = self.search0(img_nn0[None,:],img_nn0[None,:],qindex,bsize)
        nlDists = nlDists[0]


    #
    # -- Scatter Section --
    #

    # -- indexing patches --
    t,c,h,w = image_n0.shape
    patches = scatter_nl(image_n0[None,:],nlInds)[0]
    ishape = 'p k 1 c h w -> 1 p k (c h w)'
    patches = rearrange(patches,ishape)


    # -- append anchor patch spatial variance --
    d = patches.shape[-1]
    pvar0 = patches[0,:,0,:].std(-1)**2*d # patch var
    nlDists = th.cat([nlDists[:,1:],pvar0[:,None]],-1)

    # -- remove padding --
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    # -- view --
    # print(nlInds.shape)
    # print(nlInds[-132*2-64,:])
    # print(nlInds[-64,:])
    # print(nlInds.shape)
    # print(queryInds[-132*2-64,:])
    # print(queryInds[-64,:])
    # print(nlDists[-132*2-64,:])
    # print(nlDists[-64,:])

    # -- pack up --
    info = edict()
    info.patches = patches
    info.dists = nlDists[None,:]
    return info


@register_method
def run_nn1(self,image_n,queryInds,scatter_nl,
            srch_img=None,flows=None,train=False,ws=29,wt=0,
            neigh_pad = 14, detach_search = True, bsize=-1):

    # -- unpack --
    t = image_n.shape[0]
    ps = self.ps#patch_w
    device = image_n.device
    patch_numel = (ps ** 2) * image_n.shape[1]

    # -- nugber of patches along (height,width) --
    t,c,h,w = image_n.shape
    vshape = image_n.shape
    hp,wp = get_npatches(vshape, train, self.ps, self.pad_offs, self.k, self.lidia_pad)

    # -- pad image --
    image_n1 = self.prepare_image_n1(image_n,train)
    _,_,ha,wa = image_n1.shape
    sh,sw = (ha-hp)//2,(wa-wp)//2
    params = None

    #
    #  -- DNLS Search --
    #

    # -- get search image --
    if not(srch_img is None):
        img_nn1 = self.prepare_image_n1(srch_img,train)
        img_nn1 = self.rgb2gray(img_nn1[:,:3])
    elif self.arch_opt.rgb:
        img_nn1 = self.rgb2gray(image_n1[:,:3])
    else:
        img_nn1 = image_n1
    if detach_search:
        img_nn1 = img_nn1.detach()

    # -- print for testing --
    self.print_gpu_stats("NN1")

    # -- inds offsets --
    t,c,h0,w0 = image_n1.shape

    # -- exec search --
    k,pt,chnls = 14,1,1
    if self.lidia_pad:
        # -- padding --
        queryInds[...,1] += sh
        queryInds[...,2] += sw

        # -- no batch flows --
        flows = flow.remove_batch(flows)

        # -- search --
        nlDists,nlInds = dnls.simple.search.run(img_nn1,queryInds,flows,
                                                k,ps,pt,ws,wt,chnls,
                                                stride=2,dilation=2,
                                                reflect_bounds=False)
        # -- remove padding --
        queryInds[...,1] -= sh
        queryInds[...,2] -= sw

    else:
        qindex = queryInds
        self.update_search_flows(self.search1,img_nn1[None,:].shape,
                                 img_nn1.device,flows)
        nlDists,nlInds = self.search1(img_nn1[None,:],img_nn1[None,:],qindex,bsize)
        nlDists = nlDists[0]

    #
    # -- Scatter Section --
    #

    # -- dnls --
    patches = scatter_nl(image_n1[None,:],nlInds)[0]

    #
    # -- Final Formatting --
    #

    # - reshape --
    ishape = 'p k 1 c h w -> 1 p k (c h w)'
    patches = rearrange(patches,ishape)

    # -- patch variance --
    d = patches.shape[-1]
    pvar0 = patches[0,:,0,:].std(-1)**2*d # patch_var
    nlDists = th.cat([nlDists[:,1:],pvar0[:,None]],-1)

    # -- centering inds --
    t,c,h1,w1 = image_n1.shape
    sh,sw = (h1 - hp)//2,(w1 - wp)//2
    nlInds[...,1] -= sh
    nlInds[...,2] -= sw

    # -- pack up --
    info = edict()
    info.patches = patches
    info.dists = nlDists[None,:]
    return info


@register_method
def init_search(self):
    self.init_search_nn0()
    self.init_search_nn1()

@register_method
def init_search_nn0(self):
    fflow,bflow = None,None
    stride0 = 1
    stride1 = 1
    pt = 1
    ps = self.ps
    ws = self.ws
    wt = self.wt
    k = self.k
    dil = 1
    full_ws = False
    use_k = True
    exact = False
    use_adj = False
    reflect_bounds = False
    search_abs = False
    remove_self = False
    rbwd,nbwd = False,1
    self.search0 = dnls.search.init("l2_with_index",fflow, bflow, k,
                                    ps, pt, ws, wt,chnls=-1,dilation=dil,
                                    stride0=stride0,stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    search_abs=search_abs,
                                    use_k=use_k,use_adj=use_adj,
                                    full_ws=full_ws,exact=exact,
                                    remove_self=remove_self,
                                    rbwd=rbwd,nbwd=nbwd)
@register_method
def init_search_nn1(self):
    fflow,bflow = None,None
    stride0 = 1
    stride1 = 2
    dil = 2
    pt = 1
    ps = self.ps
    ws = self.ws
    wt = self.wt
    k = self.k
    full_ws = False
    use_k = True
    exact = False
    use_adj = False
    reflect_bounds = False
    search_abs = False
    remove_self = False
    rbwd,nbwd = False,1
    self.search1 = dnls.search.init("l2_with_index",fflow, bflow, k,
                                    ps, pt, ws, wt,chnls=-1,dilation=dil,
                                    stride0=stride0,stride1=stride1,
                                    reflect_bounds=reflect_bounds,
                                    search_abs=search_abs,
                                    use_k=use_k,use_adj=use_adj,
                                    full_ws=full_ws,exact=exact,
                                    remove_self=remove_self,
                                    rbwd=rbwd,nbwd=nbwd)

@register_method
def update_search_flows(self,search,vshape,device,flows):
    _flows = match_shape(vshape,flows)
    search.update_flow(vshape,device,_flows)

def match_shape(vshape,flows):
    if flows is None: return None

    # -- compute pad --
    iH,iW = vshape[-2:]
    fH,fW = flows.fflow.shape[-2:]
    padH = (iH - fH)//2
    padW = (iW - fW)//2
    assert padH*2 == (iH - fH)
    assert padW*2 == (iW - fW)

    # -- pad --
    _flows = edict()
    _flows.fflow = nn_pad(flows.fflow,[padW,padW,padH,padH],mode='constant',value=0.)
    _flows.bflow = nn_pad(flows.bflow,[padW,padW,padH,padH],mode='constant',value=0.)

    return _flows
