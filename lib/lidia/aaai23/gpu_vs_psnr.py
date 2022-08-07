# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- caching results --
import cache_io

# -- network --
import lidia

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.frame_start = 0
    cfg.frame_end = 4
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    return cfg

def load_intro(cfg):

    # -- get cache --
    lidia_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(lidia_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- add to cfg --
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 5
    cfg.ws = 7
    cfg.wt = 10
    cfg.isize = "none"
    def_cfg = default_cfg()
    cfg_l = [cfg]
    cache_io.append_configs(cfg_l,def_cfg) # merge the two
    cfg_l[0].isize = None # todo: fix this

    # -- load results --
    records = cache.load_flat_records(cfg_l)
    return records
