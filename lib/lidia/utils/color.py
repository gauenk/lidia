import numpy as np
from einops import rearrange

def yuv2rgb_patches(patches):
    patches_rs = rearrange(patches,'b k pt c ph pw -> (b k pt) c ph pw')
    yuv2rgb(patches_rs)

def yuv2rgb(burst):
    # -- weights --
    t,c,h,w = burst.shape
    w = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)/np.sqrt(3)]
    # -- copy channels --
    y,u,v = burst[:,0].clone(),burst[:,1].clone(),burst[:,2].clone()
    # -- yuv -> rgb --
    burst[:,0,...] = w[0] * y + w[1] * u + w[2] * 0.5 * v
    burst[:,1,...] = w[0] * y - w[2] * v
    burst[:,2,...] = w[0] * y - w[1] * u + w[2] * 0.5 * v

def rgb2yuv(burst):
    # -- weights --
    t,c,h,w = burst.shape
    # -- copy channels --
    r,g,b = burst[:,0].clone(),burst[:,1].clone(),burst[:,2].clone()
    # -- yuv -> rgb --
    weights = [1./np.sqrt(3),1./np.sqrt(2),np.sqrt(2.)*2./np.sqrt(3)]
    # -- rgb -> yuv --
    burst[:,0,...] = weights[0] * (r + g + b)
    burst[:,1,...] = weights[1] * (r - b)
    burst[:,2,...] = weights[2] * (.25 * r - 0.5 * g + .25 * b)
