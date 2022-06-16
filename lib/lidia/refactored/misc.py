

# -- linalg --
import torch as th
import numpy as np

# -- data mngmnt --
from easydict import EasyDict as edict

def select_sigma(sigma):
    sigmas = np.array([15, 25, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def get_default_config(sigma):
    cfg = edict()
    cfg.sigma = sigma
    cfg.seed = 123
    cfg.block_w = 64
    cfg.lr = 1e-3
    cfg.epoch_num = 2
    cfg.epochs_between_check = 1
    cfg.max_batches = 500
    cfg.dset_stride = 1
    cfg.train_batch_size = 4
    cfg.device = "cuda:0"
    return cfg

def calc_padding_ps(patch_w=5):
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + 13
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + 14 * 2
    offs = total_pad - total_pad0
    return offs,total_pad

def calc_padding_rgb(patch_w=5):
    calc_padding_ps(patch_w)

def calc_padding(arch_opt):
    patch_w = 5 if arch_opt.rgb else 7
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + 13
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + 14 * 2
    offs = total_pad - total_pad0
    return offs, total_pad

def crop_offset(in_image, row_offs, col_offs):
    if len(row_offs) == 1:
        row_offs += row_offs
    if len(col_offs) == 1:
        col_offs += col_offs
    if row_offs[1] > 0 and col_offs[1] > 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], col_offs[0]:-col_offs[-1]]
    elif row_offs[1] > 0 and col_offs[1] == 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], :]
    elif 0 == row_offs[1] and col_offs[1] > 0:
        out_image = in_image[..., :, col_offs[0]:-col_offs[1]]
    else:
        out_image = in_image
    return out_image

def get_npatches(image, patch_w, neigh_pad):
    batches = image.shape[0]
    pixels_h = image.shape[2] - 2 * neigh_pad
    pixels_w = image.shape[3] - 2 * neigh_pad
    patches_h = pixels_h - (patch_w - 1)
    patches_w = pixels_w - (patch_w - 1)
    return patches_h,patches_w

def get_image_params(image, patch_w, neigh_pad):
    # print("image.shape: ",image.shape)
    im_params = dict()
    im_params['batches'] = image.shape[0]
    im_params['pixels_h'] = image.shape[2] - 2 * neigh_pad
    im_params['pixels_w'] = image.shape[3] - 2 * neigh_pad
    im_params['patches_h'] = im_params['pixels_h'] - (patch_w - 1)
    im_params['patches_w'] = im_params['pixels_w'] - (patch_w - 1)
    im_params['patches'] = im_params['patches_h'] * im_params['patches_w']
    im_params['pad_patches_h'] = image.shape[2] - (patch_w - 1)
    im_params['pad_patches_w'] = image.shape[3] - (patch_w - 1)
    im_params['pad_patches'] = im_params['pad_patches_h'] * im_params['pad_patches_w']
    return im_params

def print_gpu_stats(gpu_stats,name):
    fmt_all = "[%s] Memory Allocated: %2.3f"
    fmt_res = "[%s] Memory Reserved: %2.3f"
    if gpu_stats:
        th.cuda.empty_cache()
        th.cuda.synchronize()
        mem = th.cuda.memory_allocated() / 1024**3
        print(fmt_all % (name,mem))
        mem = th.cuda.memory_reserved() / 1024**3
        print(fmt_res % (name,mem))

