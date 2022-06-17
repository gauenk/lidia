
# -- misc --
import os,math,tqdm
import pprint,random
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

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import lidia

# -- padding --
from torch.nn.functional import pad

def get_model(mtype,sigma,ishape,device):
    if mtype == "batched":
        model = lidia.batched.load_model(sigma,lidia_pad=True).to(device)
        model.eval()
    elif mtype == "original":
        model = lidia.refactored.load_model(sigma,"original").to(device)
        model.eval()
    else:
        raise ValueError(f"Uknown mtype [{mtype}]")
    return model

def run_model(cfg,model,noisy,flows,batch_size):
    with th.no_grad():
        deno = model(noisy,cfg.sigma,flows=flows,
                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
    deno = deno.detach()
    return deno

def run_exp(cfg):

    # -- set seed --
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- clear cache --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- create timer --
    timer = lidia.utils.timer.ExpTimer()

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    index = data.te.groups.index(cfg.vid_name)
    sample = data.te[index]

    # -- unpack --
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

    # -- modded isize --
    h,w = noisy.shape[-2:]
    msize = cfg.res
    min_edge = min(h,w)
    if msize < min_edge:
        rem = int(min_edge - msize)
        noisy = pad(noisy,[rem,]*4,mode="reflect")
        clean = pad(clean,[rem,]*4,mode="reflect")
    noisy = noisy[...,:msize,:msize].contiguous()
    clean = clean[...,:msize,:msize].contiguous()
    # print("noisy.shape: ",noisy.shape)

    # -- network --
    model = get_model(cfg.mtype,cfg.sigma,noisy.shape,cfg.device)

    # -- optical flow --
    timer.start("flow")
    if cfg.comp_flow == "true":
        noisy_np = noisy.cpu().numpy()
        flows = svnlb.compute_flow(noisy_np,cfg.sigma)
        flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
    else:
        flows = None
    timer.stop("flow")

    # -- size --
    nframes = noisy.shape[0]
    npix_f = np.prod(list(noisy.shape[-2:]))
    npix = nframes * npix_f
    if cfg.batch_perc > .999:
        batch_size = -1
    else:
        batch_size = int(npix * cfg.batch_perc)

    # -- internal adaptation --
    timer.start("adapt")
    run_internal_adapt = cfg.internal_adapt_nsteps > 0
    run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
    if run_internal_adapt:
        model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                                 ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                 nsteps=cfg.internal_adapt_nsteps,
                                 nepochs=cfg.internal_adapt_nepochs)
    timer.stop("adapt")

    # -- denoise --
    timer.start("deno")
    fail = False
    try:
        deno = run_model(cfg,model,noisy,flows,batch_size)
    except:
        fail = True
    timer.stop("deno")

    # -- save example --
    if not fail:
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = lidia.utils.io.save_burst(deno,out_dir,"deno")
    else:
        deno_fns = ["" for t in range(nframes)]

    # -- psnr --
    if not fail:
        t = clean.shape[0]
        deno = deno.detach()
        clean_rs = clean.reshape((t,-1))/255.
        deno_rs = deno.reshape((t,-1))/255.
        mse = th.mean((clean_rs - deno_rs)**2,1)
        psnrs = -10. * th.log10(mse).detach()
        psnrs = list(psnrs.cpu().numpy())
    else:
        psnrs = [-1. for t in range(nframes)]

    # -- init results --
    results = edict()
    results.psnrs = psnrs
    results.deno_fn = deno_fns
    results.vid_name = [cfg.vid_name]
    results.batch_size = int(batch_size)
    results.failed = int(fail)
    for name,time in timer.items():
        results[name] = time
    print(results)

    return results


def default_cfg():
    # -- config --
    cfg = edict()
    # cfg.nframes_tr = 5
    # cfg.nframes_val = 5
    # cfg.nframes_te = 0
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.seed = 123
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "show_stats" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    # dnames = ["toy"]
    # vid_names = ["text_tourbus"]
    dnames = ["set8"]
    vid_names = ["tractor"]
    sigmas = [30]#,50]
    internal_adapt_nsteps = [0]#,500]
    internal_adapt_nepochs = [5]
    ws,wt = [29],[0]
    # ws,wt = [10],[10]
    nframes = [1]
    comp_flow = ["false"]

    # -- sets of exps --
    mtype = ["batched"]
    batch_perc = [1.,0.9,0.8]
    res = [96,128,256,300,385,400,425,500]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt,
                 "mtype":mtype,"res":res,"nframes":nframes,
                 "batch_perc":batch_perc}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh

    mtype = ["batched"]
    batch_perc = [0.65,0.60,0.55,0.5,0.45,0.4]
    res = [500,700]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt,
                 "mtype":mtype,"res":res,"nframes":nframes,
                 "batch_perc":batch_perc}
    exps += cache_io.mesh_pydicts(exp_lists) # create mesh

    mtype = ["batched"]
    batch_perc = [0.40,0.35,0.30,0.25]
    res = [1000]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt,
                 "mtype":mtype,"res":res,"nframes":nframes,
                 "batch_perc":batch_perc}
    exps += cache_io.mesh_pydicts(exp_lists) # create mesh

    mtype = ["batched"]
    batch_perc = [0.25,0.2,0.15,0.1]
    res = [2000]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt,
                 "mtype":mtype,"res":res,"nframes":nframes,
                 "batch_perc":batch_perc}
    exps += cache_io.mesh_pydicts(exp_lists) # create mesh

    mtype = ["original"]
    batch_perc = [1.]
    res = [96,128,256,300,385,400,425]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt,
                 "mtype":mtype,"res":res,"nframes":nframes,
                 "batch_perc":batch_perc}
    exps += cache_io.mesh_pydicts(exp_lists) # create mesh


    # pp.pprint(exps)
    for exp in exps:
        if exp.internal_adapt_nsteps == 0:
            exp.internal_adapt_nepochs = 2

    # -- group with default --
    base_cfg = default_cfg()
    cache_io.append_configs(exps,base_cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # if exp.mtype == "original":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    print(records.filter(like="timer"))

    # -- print by area --
    for mtype,mdf in records.groupby("mtype"):
        print("--- Mtype [%s] ---" % mtype)
        for fail_id,fdf in mdf.groupby("failed"):
            print("--- Failed Id [%d] ---" % fail_id)
            print(fdf[['res','batch_perc','timer_deno']])

    # -- create stat plots --
    lidia.plots.show_stats.create_stat_plots(records)

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        field = "internal_adapt_nsteps"
        for adapt,adf in ddf.groupby(field):
            for cflow,fdf in adf.groupby("comp_flow"):
                for ws,wsdf in fdf.groupby("ws"):
                    for wt,wtdf in wsdf.groupby("wt"):
                        print("adapt,ws,wt,cflow: ",adapt,ws,wt,cflow)
                        for sigma,sdf in wtdf.groupby("sigma"):
                            ave_psnr,ave_time,num_vids = 0,0,0
                            for vname,vdf in sdf.groupby("vid_name"):
                                ave_psnr += vdf.psnrs.mean()
                                ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
                                num_vids += 1
                            ave_psnr /= num_vids
                            ave_time /= num_vids
                            total_frames = len(sdf)
                            fields = (sigma,ave_psnr,ave_time,total_frames)
                            print("[%d]: %2.3f @ ave %2.2f sec for %d frames" % fields)


if __name__ == "__main__":
    main()
