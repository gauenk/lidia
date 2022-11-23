# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
import copy
dcopy = copy.deepcopy

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
import lidia.explore_configs as explore_configs


def get_base_cfg():
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    return cfg

def merge_with_base(cfg):
    # -- [first] merge with base --
    cfg_og = dcopy(cfg)
    cfg_l = [cfg]
    cfg_base = get_base_cfg()
    cache_io.append_configs(cfg_l,cfg_base)
    cfg = cfg_l[0]

    # -- overwrite with input values --
    for key in cfg_og:
        cfg[key] = cfg_og[key]
    return cfg

def load_results(cfg):

    # -- load cache --
    colanet_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(colanet_home / ".cache_io")
    cache_name = "explore_grid" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- set config --
    cfg = merge_with_base(cfg)
    cfg.bw = True
    cfg.sigma = 30
    # cfg.nframes = 10
    # cfg.frame_start = 0
    # cfg.frame_end = cfg.frame_start+cfg.nframes-1
    # cfg.dname = "set8"
    # cfg.sigma = 10.
    # cfg.vid_name = "sunflower"
    cfg.internal_adapt_nsteps = 200
    cfg.internal_adapt_nepochs = 0
    cfg.flow = "true"
    cfg.adapt_mtype = "rand"
    # cfg.use_train = "true"
    cfg.model_type = "batched"
    del cfg['isize']

    # -- config grid [1/3] --
    exps_a = explore_configs.search_space_cfg()
    ws,wt = [10,15,20,25,30],[0,1,2,3,5]
    bs = [256,1024,10*1024,128*128*3]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":["128_128"]}
    # exps_a = cache_io.mesh_pydicts(exp_lists) # get grid
    # cache_io.append_configs(exps_a,cfg) # merge the two

    # -- config grid [2/3] --
    exp_lists['ws'] = [5]
    exp_lists['wt'] = [2,3,5]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- config grid [(ws,wt) vs (runtime, gpu, psnr)] --
    bs = [48*1024]
    ws,wt = [10,15,20,25,30],[5,0,1,2,3,4]
    flow = ["true"]
    isizes = ["128_128",]
    exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.internal_adapt_nepochs = 1
    cache_io.append_configs(exps_c,cfg) # merge the two

    # -- config grid [(resolution,batch_size) vs (mem,runtime)] --
    exp_lists['ws'] = [20]
    exp_lists['wt'] = [3]
    exp_lists['isize'] = ["220_220","180_180","140_140","100_100","60_60"]
    exp_lists['bs'] = [220*220*3,180*180*3,140*140*3,100*100*3,60*60*3]
    exps_d = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.nframes = 3
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    cfg.internal_adapt_nepochs = 0
    cache_io.append_configs(exps_d,cfg) # merge the two

    # -- combine --
    exps = exps_a# + exps_d + exps_c# + exps_b + exps_c
    pp.pprint(exps[-1])

    # -- read --
    root = Path("./.aaai23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "lidia_explore_grid.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=False)

    # -- standardize col names --
    records = records.rename(columns={"bs":"batch_size"})

    return records
