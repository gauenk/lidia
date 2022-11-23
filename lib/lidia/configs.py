"""
Baseline Configs

"""

from easydict import EasyDict as edict

def base_config():
    return default_cfg()

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.model_type = "batched"
    cfg.adapt_mtype = "rand"
    return cfg
