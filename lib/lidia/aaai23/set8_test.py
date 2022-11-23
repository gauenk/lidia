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
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    return cfg

def append_detailed_cfg(cfg):
    # -- add to cfg --
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 200
    cfg.internal_adapt_nepochs = 1
    # cfg.isize = "none"
    def_cfg = default_cfg()
    cfg_l = [cfg]
    cache_io.append_configs(cfg_l,def_cfg) # merge the two
    # if cfg_l[0].isize == "none": cfg_l[0].isize = None
    return cfg_l[0]

def load_proposed(cfg,ws=15,wt=5):
    return load_results("batched",ws,wt,cfg)

def load_original(cfg):
    ws = 29
    wt = 0
    cfg.flow = "false"
    return load_results("batched",ws,wt,cfg)

def load_results(mtype,ws,wt,cfg):
    # -- get cache --
    lidia_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(lidia_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cfg = append_detailed_cfg(cfg)
    cfg_l = [cfg]

    # -- assign --
    cfg.model_type = mtype
    cfg.ws = ws
    cfg.wt = wt
    cfg.seed = 123

    # -- load results --
    pp.pprint(cfg_l[0])
    records = cache.load_flat_records(cfg_l)
    records['home_path'] = lidia_home
    return records
