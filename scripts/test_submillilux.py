
# -- misc --
import os,math,tqdm
import random,pprint
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

# -- submillilux noise gen --
import stardeno

# -- network --
import lidia

# -- caching results --
import cache_io

def run_exp(cfg):

    # -- init results --
    results = edict()
    results.psnrs = []
    results.deno_fn = []
    results.names = []
    results.timer_adapt = []
    results.timer_deno = []

    # -- set seed --
    random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    loader = iter(loaders.te)

    # -- network --
    model = lidia.batched.load_model(cfg.model_sigma,lidia_pad=True).to(cfg.device)
    model.eval()

    # -- for each sample --
    for sample in loader:

        # -- create timer --
        timer = lidia.utils.timer.ExpTimer()

        # -- unpack --
        name = data.te.groups[int(sample['index'][0])]
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

        # -- size --
        npix = np.prod(noisy.shape[-2:])
        ngroups = int(npix/390.*390.)
        batch_size = ngroups#*1024
        batch_size = -1

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        if run_internal_adapt:

            # -- load noise simulator --
            noise_sim = stardeno.load_noise_sim(cfg.device)

            # -- run adaptation --
            model.run_internal_adapt(noisy,cfg.model_sigma,flows=None,
                                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                     nsteps=cfg.internal_adapt_nsteps,
                                     nepochs=cfg.internal_adapt_nepochs,
                                     noise_sim=noise_sim)
        timer.stop("adapt")

        # -- denoise --
        timer.start("deno")
        with th.no_grad():
            deno = model(noisy,cfg.model_sigma)
            # deno = model(noisy,cfg.sigma,flows=None,
            #              ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
            deno = deno.detach()
        timer.stop("deno")

        # -- save example --
        out_dir = Path(cfg.saved_dir)# / str(cfg.uuid)
        if not out_dir.exists(): out_dir.mkdir(parents=True)
        noisy_fn = out_dir / ("noisy_%s.png" % name)
        lidia.utils.io.save_image(noisy[0]/255.,noisy_fn)
        deno_fn = out_dir / ("deno_%s.png" % name)
        lidia.utils.io.save_image(deno[0],deno_fn)

        # -- psnr --
        noisy_psnr = -10. * th.log10(th.mean((clean/255. - noisy/255.)**2)).item()
        # print(noisy_psnr)
        psnr = -10. * th.log10(th.mean((clean/255. - deno/255.)**2)).item()
        # print(psnr)
        del noisy,clean
        del sample
        th.cuda.empty_cache()

        # -- init results --
        results.psnrs.append(psnr)
        results.deno_fn.append(deno_fn)
        results.names.append([name])
        for name,time in timer.items():
            results[name].append(time)

    return results

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.nframes = 1
    cfg.saved_dir = "./output/saved_results/batched_lidia/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.seed = 123
    # cfg.isize = "256_256"
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_img" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    dnames = ["submillilux"]
    internal_adapt_nsteps = [0]#,500]
    internal_adapt_nepochs = [5]
    sigmas = [50.]
    ws,wt = [29],[0]
    exp_lists = {"dname":dnames,"model_sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "ws":ws,"wt":wt}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    # pp.pprint(exps)

    # -- group with default --
    cfg = default_cfg()
    cache_io.append_configs(exps,cfg) # merge the two

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
        # cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    print(records.filter(like="timer"))

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        field = "internal_adapt_nsteps"
        for adapt,adf in ddf.groupby(field):
            for ws,wsdf in adf.groupby("ws"):
                for wt,wtdf in wsdf.groupby("wt"):
                    print("adapt,ws,wt: ",adapt,ws,wt)
                    psnr = wtdf['psnr']
                    time = wtdf['timer_deno']
                    fields = (psnr,time)
                    print("%2.3f for %2.2f" % fields)


if __name__ == "__main__":
    main()
