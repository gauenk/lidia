
# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- vision/shaping --
import torch.nn.functional as nn_func
from torch.nn.functional import pad as nn_pad

# -- diff. non-local search --
import dnls

# -- separate class and logic --
from n4net.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)

# -- helper imports --
from n4net.utils.inds import get_3d_inds
from .misc import get_image_params,crop_offset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Shaping/Pads for Fold/Unfold/Scatter
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def image_shape(self, ishape, ps, dilation=1, train=False):
    h,w = ishape
    pad = dilation*(ps//2)
    hp,wp = h+2*pad,w+2*pad
    return hp,wp,pad

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#          Shapes for Image 0
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def pad_crop0(self, image, pad_offs, train):
    if self.lidia_pad:
        return self._pad_crop0_og(image, pad_offs, train, self.ps)
    else:
        return self._pad_crop0_eff(image, pad_offs, train, self.ps)

@register_method
def _pad_crop0_eff(self,image,pad_offs,train,ps):
    pad = ps//2
    image_n0 = nn_pad(image,[pad,]*4,mode="reflect")
    return image_n0

@register_method
def _pad_crop0_og(sel,image,pad_offs,train,ps):
    if not train:
        reflect_pad = [ps - 1] * 4
        constant_pad = [14] * 4
        image = nn_func.pad(nn_func.pad(image, reflect_pad, 'reflect'),
                            constant_pad, 'constant', -1)
    else:
        image = crop_offset(image, (pad_offs,), (pad_offs,)).contiguous()
    return image

# @register_method
# def image_n0_shape(self,image_n,train,k=14):
#     if self.lidia_pad:
#         return self.image_n0_shape_og(image_n,train,k=k)
#     else:
#         pad = self.ps//2
#         t,c,h,w = image_n.shape
#         hp,wp = h+2*pad,w+2*pad
#         return t,c,hp,wp

# @register_method
# def image_n0_shape_og(self,image_n,train,k=14):
#     pad_offs = self.pad_offs
#     ps = self.patch_w
#     if not train:
#         reflect_pad = ps - 1
#         constant_pad = k
#         pad = reflect_pad + constant_pad
#         t,c,h,w = image_n.shape
#         hp = h + 2*pad
#         wp = w + 2*pad
#         return t,c,hp,wp
#     else:
#         t,c,h,w = image_n.shape
#         hp = h - pad_offs
#         wp = w - pad_offs
#         return t,c,hp,wp


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#          Shapes for Image 1
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@register_method
def pad_crop1(self, image, train, mode):
    return self._pad_crop1(image, train, mode, self.patch_w)

@register_method
def _pad_crop1(self, image, train, mode, patch_w):
    if not train:
        if mode == 'reflect':
            bilinear_pad = 1
            averaging_pad = (patch_w - 1) // 2
            patch_w_scale_1 = 2 * patch_w - 1
            find_nn_pad = (patch_w_scale_1 - 1) // 2
            reflect_pad = [averaging_pad + bilinear_pad + find_nn_pad] * 4
            image = nn_func.pad(image, reflect_pad, 'reflect')
        elif mode == 'constant':
            constant_pad = [28] * 4
            # image = nn_func.pad(image, constant_pad, 'reflect')
            image = nn_func.pad(image, constant_pad, 'constant', -1)
        else:
            assert False
    return image

# @register_method
# def image_n1_shape(self,image_n,train,k=14):
#     if self.lidia_pad:
#         return self.image_n1_shape_og(image_n,self.ps,train,k)
#     else:
#         return self.image_n1_shape_eff(image_n,self.ps,train,k)

# @register_method
# def image_n1_shape_eff(self,image_n,ps,train,k=14):
#     t,c,h,w = image_n.shape
#     hp,wp = h+2*(ps//2),w+2*(ps//2)
#     return t,c,hp,wp

# @register_method
# def image_n1_shape_og(self,image_n,ps,train,k=14):
#     if train:
#         t,c,h,w = image_n.shape
#         hp,wp = h-2,w-2
#         return t,c,h,w
#     else:
#         bilinear_pad = 1
#         averaging_pad = (ps - 1) // 2
#         ps_scale_1 = 2 * ps - 1
#         find_nn_pad = (ps_scale_1 - 1) // 2
#         constant_pad = 2*k
#         pad = averaging_pad + find_nn_pad + constant_pad
#         t,c,h,w = image_n.shape
#         hp,wp = h+2*pad,w+2*pad
#         return t,c,hp,wp

@register_method
def prepare_image_n1(self,image_n,train):
    if self.lidia_pad:
        return self.prepare_image_n1_og(image_n,train)
    else:
        return self.prepare_image_n1_eff(image_n)

@register_method
def prepare_image_n1_eff(self,image_n):
    """
    This version of prep. does not include extra padding
    """
    # -- binilear conv --
    pad = self.ps//2
    image_n1 = nn_pad(image_n,[pad+1,]*4,mode="reflect")
    t,c,h,w = image_n1.shape
    image_n1 = image_n1.view(t*c,1,h,w)
    image_n1 = self.bilinear_conv(image_n1)
    image_n1 = image_n1.view(t,c,h-2,w-2) # remove 1 each side
    return image_n1

@register_method
def prepare_image_n1_og(self,image_n,train):
    """
    The original version includes extra padding for differentiability
    """
    # -- pad & unpack --
    patch_numel = (self.patch_w ** 2) * image_n.shape[1]
    device = image_n.device
    image_n1 = self.pad_crop1(image_n, train, 'reflect')

    # -- bilinear conv --
    # image_n1 = image_n1[:,:,1:-1,1:-1]
    im_n1_b, im_n1_c, im_n1_h, im_n1_w = image_n1.shape
    image_n1 = image_n1.view(im_n1_b * im_n1_c, 1,im_n1_h, im_n1_w)
    image_n1 = self.bilinear_conv(image_n1)
    image_n1 = image_n1.view(im_n1_b, im_n1_c, im_n1_h - 2, im_n1_w - 2)

    # -- final pad --
    image_n1 = self.pad_crop1(image_n1, train, 'constant')
    return image_n1
