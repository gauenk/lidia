
from . import configs
import cache_io

def modulation_cfg():

    # -- config --
    cfg = configs.base_config()
    cfg.internal_adapt_nsteps = 200
    cfg.internal_adapt_nepochs = 0
    cfg.flow = "true"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.sigma = 30
    cfg.dname = "set8"
    cfg.vid_name = "sunflower"

    # -- meshgrid --
    ws,wt = [20],[0]
    bs = [256,1024,10*1024,128*128*3]
    isize = ["128_128"]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps = cache_io.mesh_pydicts(exp_lists)
    cache_io.append_configs(exps,cfg)

    return exps

def search_space_cfg():


    # -- baseline --
    cfg = configs.base_config()
    cfg.internal_adapt_nsteps = 200
    cfg.internal_adapt_nepochs = 1
    cfg.flow = "true"
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.sigma = 30
    cfg.dname = "set8"
    cfg.vid_name = "sunflower"

    # -- our method's grid --
    bs = [24*1024]
    wt = [5,0,1,2,3,4]
    ws = [10,15,20,25,30]
    isize = ["128_128"]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps_0 = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_0,cfg)

    # -- original method for reference  --
    T = cfg.nframes
    bs,ws,wt = [128*128*T],[20],[0]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps_1 = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.flow = "false"
    cfg.model_name = "batched"
    cache_io.append_configs(exps_1,cfg)

    # -- combine --
    exps = exps_0 + exps_1

    return exps


def resolution_cfg():

    # -- baseline --
    cfg = configs.base_config()
    cfg.internal_adapt_nsteps = 200
    cfg.internal_adapt_nepochs = 0
    cfg.flow = "true"
    cfg.nframes = 3
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.sigma = 30
    cfg.dname = "set8"
    cfg.vid_name = "sunflower"


    # -- meshgrid --
    ws,wt = [20],[3]
    bs = [220*220*3,180*180*3,140*140*3,100*100*3,60*60*3]
    isize = ["220_220","180_180","140_140","100_100","60_60",]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps = cache_io.mesh_pydicts(exp_lists)
    cache_io.append_configs(exps,cfg)

    return exps

