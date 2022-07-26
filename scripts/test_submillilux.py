
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

# -- optical flow --
import svnlb

# -- data --
import data_hub

# -- submillilux noise gen --
import stardeno
from stardeno import pp as spp

# -- network --
import lidia

# -- caching results --
import cache_io

def run_exp(cfg):

    # -- init results --
    results = edict()
    results.deno_fns = []
    results.pp_deno_fns = []
    results.vid_name = []
    results.vid_frames = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []

    # -- set seed --
    random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- network --
    model = lidia.batched.load_model(cfg.model_sigma,lidia_pad=False).to(cfg.device)
    model.eval()
    fwd = model.forward
    def rslice(vid,coords):
        if coords is None: return vid
        fs,fe,t,l,b,r = coords
        return vid[fs:fe,:,t:b,l:r]
    def wrap_fwd(noisy,sigma, srch_img=None, flows=None,
                 ws=29, wt=0, train=False, rescale=True, stride=1,
                 batch_size = -1, batch_alpha = 0.5, region=None):
        noisy_3d = noisy[:,:3].contiguous()
        deno_3d = fwd(noisy_3d,sigma,srch_img,flows,ws,wt,train,rescale,
                      stride,batch_size,batch_alpha,region)
        depth_1d = rslice(noisy[:,[3]],region)
        deno_4d = th.cat([deno_3d,depth_1d],1)
        return deno_4d

    model.forward = wrap_fwd

    # -- get all indices with vid_name [allows for sub-sequences --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    def fbnds(fnums,lb,ub): return (lb <= np.max(fnums)) and (ub >= np.min(fnums))
    indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
                                           cfg.frame_start,cfg.frame_end)]

    # -- for each subsequence in burst --
    for index in indices:

        # -- create timer --
        timer = lidia.utils.timer.ExpTimer()

        # -- unpack sample --
        sample = data.te[index]
        name = data.te.groups[int(sample['index'][0])]
        noisy = 255.*th.pow(sample['noisy'],1./2.2)
        # noisy = 255.*noisy
        vid_frames = sample['fnums']
        noisy = noisy.to(cfg.device)
        # noisy = noisy.to(cfg.device)[:,:3].contiguous()

        # -- break early --
        run_burst = cfg.frame_start <= np.min(vid_frames)
        run_burst = run_burst and (cfg.frame_end+cfg.nframes) >= np.max(vid_frames)
        if not run_burst: continue

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            noisy_np = noisy.cpu().numpy()
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- size --
        npix = np.prod(noisy.shape[-2:])
        ngroups = int(npix/390.*390.)
        # batch_size = ngroups#*1024
        batch_size = 39*39#390*390

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        if run_internal_adapt:

            # -- load noise simulator --
            noise_sim = stardeno.load_noise_sim(cfg.device).to(cfg.device)
            def wrap_sim(clean_i):
                clean_z = 255.*((clean_i * 0.5) + 0.5)
                clean_z = th.clamp(clean_z,0.,255.)
                with th.no_grad():
                    noisy_z = noise_sim(clean_z)
                noisy_i = (noisy_z/255. - 0.5)/0.5
                return noisy_i

            # -- run adaptation --
            model.run_internal_adapt(noisy.clone(),cfg.model_sigma,flows=flows,
                                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                     nsteps=cfg.internal_adapt_nsteps,
                                     nepochs=cfg.internal_adapt_nepochs,
                                     noise_sim=wrap_sim)
        timer.stop("adapt")

        # -- denoise --
        batch_size = 390*39
        timer.start("deno")
        with th.no_grad():
            # deno = model(noisy,cfg.model_sigma)
            deno = model(noisy,cfg.model_sigma,flows=flows,
                         ws=cfg.ws,wt=cfg.wt,batch_size=batch_size)
            deno = deno.detach()
        timer.stop("deno")
        print("deno.shape: ",deno.shape)
        print(deno.max(),deno.min())
        def rescale_deno(vid):
            vid -= vid.min()
            vid /= vid.max()
            vid *= 255.
            return vid
        # deno = rescale_deno(deno)
        deno = th.clamp(deno,0.,255.)
        print(deno.max(),deno.min())

        # -- post-process --
        # pp_noisy = spp.default_process(noisy/255.,demosaic=False)
        # pp_deno = spp.default_process(deno/255.,demosaic=True)
        print("a.")
        pp_noisy = spp.default_process(th.pow(noisy/255.,2.2),demosaic=False)
        print("b.")
        pp_deno = spp.default_process(th.pow(deno/255.,2.2),demosaic=True)
        print("c.")
        # plist = [spp.bayer_bilinear, spp.white_balance, spp.clip, spp.gamma]
        # pp_deno = rearrange(deno/255.,'t c h w -> t h w c').cpu().numpy()
        # # pp_deno = np.power(pp_deno,2.2)
        # pp_deno = spp.process(pp_deno,plist)
        # if pp_deno.ndim == 3: pp_deno = pp_deno[None,:]
        # pp_deno = rearrange(pp_deno,'t h w c -> t c h w')
        # pp_deno = exposure.equalize_hist(pp_deno)

        # -- save example --
        fstart = np.min(vid_frames)
        SAVE_DIR = Path(cfg.saved_dir)
        noisy_fns = lidia.utils.io.save_burst(pp_noisy,SAVE_DIR,"noisy",fstart)
        deno_fns = lidia.utils.io.save_burst(deno,SAVE_DIR,"deno",fstart)
        print("deno.shape: ",deno.shape)
        print("pp_deno.shape: ",pp_deno.shape)
        pp_deno_fns = lidia.utils.io.save_burst(pp_deno,SAVE_DIR,"pp_deno",fstart)

        # -- append results --
        results.deno_fns.append(deno_fns)
        results.pp_deno_fns.append(pp_deno_fns)
        results.vid_name.append([cfg.vid_name])
        results.vid_frames.append(vid_frames)
        for name,time in timer.items():
            results[name].append(time)

    return results

def default_cfg():
    # 59 is the "deno" one
    # 63 is the "raw" one
    # -- config --
    cfg = edict()
    cfg.nframes = 5
    cfg.frame_start = 57
    cfg.frame_end = 66
    cfg.fskip = 1
    cfg.saved_dir = "./output/saved_results/batched_lidia/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.seed = 123
    cfg.isize = "512_512"
    # cfg.isize = "96_96"
    return cfg

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_submillilux_real" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get mesh --
    dnames = ["submillilux"]
    internal_adapt_nsteps = [5]#,500]
    internal_adapt_nepochs = [5]
    sigmas = [50.]
    ws,wt = [10],[5]
    # ws,wt = [29],[0]
    flow = ["false"]
    exp_lists = {"dname":dnames,"model_sigma":sigmas,
                 "vid_name":["seq8"],"ws":ws,"wt":wt,"flow":flow,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs}
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
                    time = wtdf['timer_deno'].mean()
                    fields = (time)
                    print("%2.3f" % fields)


if __name__ == "__main__":
    main()
