import dnls
import torch as th
import torch.nn as nn
from einops import rearrange

def get_search(k,ps,ws,wt,stride0,stride1):
    pt = 1
    oh0,ow0,oh1,ow1 = 0,0,0,0
    dil = 1
    use_k = True
    reflect_bounds = True
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nbwd = 1
    rbwd,exact = False,False
    chnls = 1
    search = dnls.search.init("l2_with_index", fflow, bflow,
                              k, ps, pt, ws, wt, chnls=1,
                              dilation=dil, stride0=stride0,stride1=stride1,
                              h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact)
    return search

def init_from_cfg(cfg):
    return NLSearch(cfg.k,cfg.ps,cfg.ws,cfg.wt,cfg.stride0,cfg.stride1)

class NLSearch():

    def __init__(self, k=7, ps=7, ws=8, wt=1, stride0=4,stride1=1):
        self.k = k
        self.ps = ps
        self.ws = ws
        self.search = get_search(k,ps,ws,wt,stride0,stride1)

    def __call__(self,vid,*args):
        B,T,C,H,W = vid.shape
        dists,inds = self.search(vid,vid)
        return dists,inds

    def flops(self,B,C,H,W):
        return self.search.flops(B,C,H,W)

    def radius(self,*args,**kwargs):
        return self.ws
