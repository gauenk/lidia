# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- diff. non-local search --
import dnls

# -- separate class and logic --
from lidia.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -- helper imports --
from lidia.utils.inds import get_3d_inds
from torch.nn.functional import conv2d,unfold
from .misc import get_image_params


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Run Nearest Neighbors Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
# def run_nn0(self,image_n,train=False):
def run_nn0(self,image_n,srch_img=None,flows=None,train=False,ws=29,wt=0):

    # -- pad & unpack --
    neigh_pad = 14
    t_n,c_n,h_n,w_n = image_n.shape
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n0 = self.pad_crop0(image_n, self.pad_offs, train)

    # -- get search image --
    if self.arch_opt.rgb: img_nn0 = self.rgb2gray(image_n0)
    else: img_nn0 = image_n0

    # -- get image-based parameters --
    params = get_image_params(image_n0, self.patch_w, neigh_pad)
    params['pad_patches_w_full'] = params['pad_patches_w']

    # -- run knn search --
    top_dist0, top_ind0 = self.find_nn(img_nn0, params, self.patch_w)

    # -- prepare [dists,inds] --
    ip = params['pad_patches']
    patch_dist0 = top_dist0.view(top_dist0.shape[0], -1, neigh_pad)[:, :, 1:]
    top_ind0 += ip * th.arange(top_ind0.shape[0],device=device).view(-1, 1, 1, 1)

    # -- get all patches -
    patches = unfold(image_n0, (self.patch_w, self.patch_w)).\
        transpose(1, 0).contiguous().view(patch_numel, -1).t()

    # -- organize patches by knn --
    patches = patches[top_ind0.view(-1), :].\
        view(top_ind0.shape[0], -1, neigh_pad, patch_numel)

    # -- append anchor patch spatial variance --
    patch_var0 = patches[:, :, 0, :].std(dim=-1).\
        unsqueeze(-1).pow(2) * patch_numel
    patch_dist0 = th.cat((patch_dist0, patch_var0), dim=-1)

    # -- convert to 3d inds --
    t,c,h,w = image_n0.shape
    ps = self.patch_w
    ch,cw = h-(ps-1),w-(ps-1)
    k = top_ind0.shape[-1]
    inds3d = get_3d_inds(top_ind0.view(-1,1,1,k),ch,cw)

    # -- rescale inds --
    inds3d[...,1] -= neigh_pad
    inds3d[...,2] -= neigh_pad

    # -- format [dists,inds] --
    h,w = params['patches_h'],params['patches_w']
    patches = rearrange(patches,'t (h w) k d -> t h w k d',h=h)
    patch_dist0 = rearrange(patch_dist0,'t (h w) k -> t h w k',h=h)
    inds3d = rearrange(inds3d,'(t h w) k tr -> t h w k tr',t=t,h=h)

    return patches,patch_dist0,inds3d,params

@register_method
# def run_nn1(self,image_n,train=False):
def run_nn1(self,image_n,srch_img=None,flows=None,train=False,ws=29,wt=0):

    # -- misc unpack --
    neigh_pad = 14
    ps = self.patch_w

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

    # -- img-based parameters --
    params = get_image_params(image_n1, 2 * self.patch_w - 1, 28)
    params['pad_patches_w_full'] = params['pad_patches_w']

    # -- get search image  --
    if self.arch_opt.rgb: img_nn1 = self.rgb2gray(image_n1)
    else: img_nn1 = image_n1

    # -- spatially split image --
    img_nn1_00 = img_nn1[:, :, 0::2, 0::2].clone()
    img_nn1_10 = img_nn1[:, :, 1::2, 0::2].clone()
    img_nn1_01 = img_nn1[:, :, 0::2, 1::2].clone()
    img_nn1_11 = img_nn1[:, :, 1::2, 1::2].clone()

    # -- get split image parameters --
    params_00 = get_image_params(img_nn1_00, self.patch_w, neigh_pad)
    params_10 = get_image_params(img_nn1_10, self.patch_w, neigh_pad)
    params_01 = get_image_params(img_nn1_01, self.patch_w, neigh_pad)
    params_11 = get_image_params(img_nn1_11, self.patch_w, neigh_pad)
    params_00['pad_patches_w_full'] = params['pad_patches_w']
    params_10['pad_patches_w_full'] = params['pad_patches_w']
    params_01['pad_patches_w_full'] = params['pad_patches_w']
    params_11['pad_patches_w_full'] = params['pad_patches_w']

    # -- run knn search! --
    top_dist1_00, top_ind1_00 = self.find_nn(img_nn1_00, params_00,
                                             self.patch_w, scale=1, case='00')
    top_dist1_10, top_ind1_10 = self.find_nn(img_nn1_10, params_10,
                                             self.patch_w, scale=1, case='10')
    top_dist1_01, top_ind1_01 = self.find_nn(img_nn1_01, params_01,
                                             self.patch_w, scale=1, case='01')
    top_dist1_11, top_ind1_11 = self.find_nn(img_nn1_11, params_11,
                                             self.patch_w, scale=1, case='11')

    # -- aggregate results [dists] --
    top_dist1 = th.zeros(params['batches'], params['patches_h'],
                            params['patches_w'], neigh_pad, device=device)
    top_dist1 = top_dist1.fill_(float('nan'))
    top_dist1[:, 0::2, 0::2, :] = top_dist1_00
    top_dist1[:, 1::2, 0::2, :] = top_dist1_10
    top_dist1[:, 0::2, 1::2, :] = top_dist1_01
    top_dist1[:, 1::2, 1::2, :] = top_dist1_11

    # -- aggregate results [inds] --
    ipp = params['pad_patches']
    top_ind1 = ipp * th.ones(top_dist1.shape,dtype=th.int64,device=device)
    top_ind1[:, 0::2, 0::2, :] = top_ind1_00
    top_ind1[:, 1::2, 0::2, :] = top_ind1_10
    top_ind1[:, 0::2, 1::2, :] = top_ind1_01
    top_ind1[:, 1::2, 1::2, :] = top_ind1_11
    top_ind1 += ipp * th.arange(top_ind1.shape[0],device=device).view(-1, 1, 1, 1)

    # -- get all patches --
    im_patches_n1 = unfold(image_n1, (self.patch_w, self.patch_w),
                           dilation=(2, 2)).transpose(1, 0).contiguous().\
                           view(patch_numel, -1).t()

    # -- organize by knn --
    np = top_ind1.shape[0]
    pn = patch_numel
    im_patches_n1 = im_patches_n1[top_ind1.view(-1), :].view(np, -1, neigh_pad, pn)

    # -- append anchor patch spatial variance --
    patch_dist1 = top_dist1.view(top_dist1.shape[0], -1, neigh_pad)[:, :, 1:]
    patch_var1 = im_patches_n1[:, :, 0, :].std(dim=-1).unsqueeze(-1).pow(2) * pn
    patch_dist1 = th.cat((patch_dist1, patch_var1), dim=-1)

    #
    # -- Final Formatting --
    #

    # -- inds -> 3d_inds --
    pad = 2*(ps//2) # dilation "= 2"
    _t,_c,_h,_w = image_n1.shape
    hp,wp = _h-2*pad,_w-2*pad
    top_ind1 = get_3d_inds(top_ind1,hp,wp)

    # -- rescale inds --
    top_ind1[...,1] -= 28
    top_ind1[...,2] -= 28

    # -- re-shaping --
    t,h,w,k = top_dist1.shape
    pdist = rearrange(patch_dist1,'t (h w) k -> t h w k',h=h)
    top_ind1 = rearrange(top_ind1,'(t h w) k tr -> t h w k tr',t=t,h=h)
    ip1 = im_patches_n1
    ip1 = rearrange(ip1,'t (h w) k d -> t h w k d',h=h)

    return ip1,pdist,top_ind1,params

@register_method
def find_nn(self, image_pad, im_params, patch_w, scale=0,
            case=None, img_search=None):
    neigh_patches_w = 2 * 14 + 1
    top_dist = th.zeros(im_params['batches'], im_params['patches_h'],
                           im_params['patches_w'], neigh_patches_w ** 2,
                           device=image_pad.device).fill_(float('nan'))
    dist_filter = th.ones(1, 1, patch_w, patch_w, device=image_pad.device)
    top_dist[...] = float("inf")

    if img_search is None: img_search = image_pad
    y = image_pad[:, :, 14:14 + im_params['pixels_h'],
                  14:14 + im_params['pixels_w']]
    for row in range(neigh_patches_w):
        for col in range(neigh_patches_w):
            lin_ind = row * neigh_patches_w + col
            y_shift = image_pad[:, :,row:row + im_params['pixels_h'],
                                col:col + im_params['pixels_w']]
            top_dist[:, :, :, lin_ind] = conv2d(((y_shift - y)) ** 2,\
                                                dist_filter).squeeze(1)

    top_dist, top_ind = th.topk(top_dist, 14, dim=3, largest=False, sorted=True)
    top_ind_rows = top_ind // neigh_patches_w
    top_ind_cols = top_ind % neigh_patches_w
    col_arange = th.arange(im_params['patches_w'], device=image_pad.device).view(1, 1, -1, 1)
    row_arange = th.arange(im_params['patches_h'], device=image_pad.device).view(1, -1, 1, 1)
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
    top_ind = top_ind_rows * im_params['pad_patches_w_full'] + top_ind_cols

    return top_dist, top_ind

