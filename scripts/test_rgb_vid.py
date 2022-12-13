
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

# -- optical flow --
# import svnlb
# from lidia import flow
from dev_basics import flow

# -- caching results --
import cache_io

# -- network --
import lidia
from lidia.utils.misc import rslice,optional,get_region_gt,set_seed
import lidia.utils.gpu_mem as gpu_mem
from lidia.utils.metrics import compute_psnrs
from lidia.utils.metrics import compute_ssims

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.noisy_psnrs = []
    results.psnrs = []
    results.ssims = []
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.mem_res = []
    results.mem_alloc = []
    results.adapt_res = []
    results.adapt_alloc = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []

    # -- network --
    # model = lidia.load_model(cfg.model_type,cfg.sigma).to(cfg.device)
    model = lidia.load_model(cfg).to(cfg.device)
    model.eval()
    imax = 255.

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,frame_start,frame_end)
    # print(th.cuda.memory_summary(abbreviated=True))

    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)

        # -- first three channels only --
        # noisy = noisy[...,:3,:,:].contiguous()
        # clean = clean[...,:3,:,:].contiguous()

        # -- create timer --
        timer = lidia.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]

        # -- optical flow --
        timer.start("flow")
        # if cfg.flow == "true":
        #     sigma_est = flow.est_sigma(noisy)
        #     # flows = flow.run(noisy,sigma_est,"svnlb")
        #     flows = flow.run(noisy,sigma_est,"cv2")
        # else:
        #     t,c,h,w = noisy.shape
        #     zflow = th.zeros((t,2,h,w),device=cfg.device,dtype=th.float32)
        #     flows = edict()
        #     flows.fflow = zflow
        #     flows.bflow = zflow
        flows = flow.orun(noisy,cfg.flow)
        print(flows.fflow.shape)
        timer.stop("flow")

        # -- info --
        print("fflow[min,max]: ",flows.fflow.min().item(),flows.fflow.max().item())
        print("bflow[min,max]: ",flows.bflow.min().item(),flows.bflow.max().item())

        # -- internal adaptation --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        timer.start("adapt")
        run_adapt = cfg.internal_adapt_nsteps > 0
        run_adapt = run_adapt and (cfg.internal_adapt_nepochs > 0)
        adapt_psnrs = [0.]
        if run_adapt:
            noisy_a = noisy[:5]
            clean_a = clean[:5]
            flows_a = flow.slice_at(flows,slice(0,5),1)
            region_gt = get_region_gt(noisy_a.shape)
            adapt_psnrs = model.run_internal_adapt(
                noisy_a,cfg.sigma,flows=flows,
                clean_gt = clean_a,region_gt = region_gt)
        timer.stop("adapt")
        adapt_alloc,adapt_res = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- denoise --
        timer.start("deno")
        with th.no_grad():
            deno = model(noisy,flows=flows)
        timer.stop("deno")
        mem_alloc,mem_res = model.mem_alloc,model.mem_res
        # mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = lidia.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        psnrs = compute_psnrs(deno,clean,div=imax)
        ssims = compute_ssims(deno,clean,div=imax)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.adapt_res.append([adapt_res])
        results.adapt_alloc.append([adapt_alloc])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results


def compute_ssim(clean,deno,div=255.):
    nframes = clean.shape[0]
    ssims = []
    for t in range(nframes):
        clean_t = clean[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        deno_t = deno[t].cpu().numpy().squeeze().transpose((1,2,0))/div
        ssim_t = compute_ssim_ski(clean_t,deno_t,channel_axis=-1)
        ssims.append(ssim_t)
    ssims = np.array(ssims)
    return ssims

def compute_psnr(clean,deno,div=255.):
    t = clean.shape[0]
    deno = deno.detach()
    clean_rs = clean.reshape((t,-1))/div
    deno_rs = deno.reshape((t,-1))/div
    mse = th.mean((clean_rs - deno_rs)**2,1)
    psnrs = -10. * th.log10(mse).detach()
    psnrs = psnrs.cpu().numpy()
    return psnrs

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/lidia/output/checkpoints/"
    # cfg.isize = "none"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    # cfg.isize = "512_512"
    # cfg.isize = "256_256"
    # cfg.isize = "128_128"
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net"
    cache_name = "test_rgb_net_supflow"
    cache_name = "test_rgb_net_svnlb"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get mesh --
    # dnames = ["toy"]
    # vid_names = ["text_tourbus"]
    # mtypes = ["rand"]
    mtypes = ["rand"]#,"sobel"]
    dnames = ["set8"]
    # vid_names = ["sunflower","snowboard","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    # vid_names = ["sunflower","snowboard"]
    # vid_names = ["tractor","motorbike"]
    # vid_names = ["hypersmooth","park_joy"]
    # vid_names = ["rafting","touchdown"]

    dnames = ["davis"]
    vid_names = ["dog"]
    # vid_names = ["bike-packing", "blackswan", "bmx-trees", "breakdance",
    #              "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
    #              "drift-chicane", "drift-straight", "goat",]
    # vid_names += ["gold-fish", "horsejump-high", "india", "judo", "kite-surf",
    #              "lab-coat", "libby", "loading", "mbike-trick",
    #              "paragliding-launch", "parkour", "pigs", "scooter-black",
    #              "shooting", "soapbox"]
    # vid_names = ["bike-packing", "blackswan", "bmx-trees", "breakdance",
    #              "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl",
    #              "dog", "dogs-jump", "drift-chicane", "drift-straight", "goat",]
    # vid_names += ["gold-fish", "horsejump-high", "india", "judo", "kite-surf",
    #              "lab-coat", "libby", "loading", "mbike-trick", "motocross-jump",
    #              "paragliding-launch", "parkour", "pigs", "scooter-black",
    #              "shooting", "soapbox"]

    # vid_names = ["tractor"]
    # sigmas = [50,30,10]#,30,10]
    sigmas = [30,50]#,30,10]
    internal_adapt_nsteps = [200]
    internal_adapt_nepochs = [1]
    # ws,wt = [15],[5]
    ws,wt = [29],[3]
    flow = [False]
    isizes = [None]#,"512_512","256_256"]
    # isizes = ["156_156"]#"256_256"]
    model_type = ["batched"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":mtypes,
                 "isize":isizes,"model_type":model_type}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- alt search --
    exp_lists['ws'] = [29]
    exp_lists['wt'] = [0]
    exp_lists['flow'] = [False]
    # exp_lists['model_type'] = ["refactored"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    exps = exps_a# + exps_b

    # -- defaults --
    # exp_lists['ws'] = [29]
    # exp_lists['wt'] = [0]
    # exp_lists['flow'] = ["false"]
    # exp_lists['model_type'] = ["refactored"]
    # exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    # exps = exps_b# + exps_a

    # pp.pprint(exps)
    # for exp in exps:
    #     if exp.internal_adapt_nsteps == 0:
    #         exp.internal_adapt_nepochs = 2

    # -- group with default --
    cfg = default_cfg()
    cfg.dset = "val"
    cfg.seed = 123
    # cfg.ntype = "submillilux"
    # cfg.isize = "256_256"
    cfg.isize = "none"
    cfg.nframes = 0
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    cfg.pretrained_load = True
    cfg.bs = 30
    cfg.bs_te = 30
    # cfg.pretrained_path = "model_state_sigma_15_c.pt"
    # cfg.pretrained_path = "model_state_sigma_25_c.pt"
    cfg.pretrained_path = "model_state_sigma_50_c.pt"
    cache_io.append_configs(exps,cfg) # merge the two
    # pp.pprint(exps[0])
    # exit(0)


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
        # if not(results is None) and len(results['psnrs']) == 0:
        #     results = None
        #     cache.clear_exp(uuid)
        # if exp.flow == "true" and results:
        #     cache.clear_exp(uuid)
        if exp.vid_name == "dog":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    print(records.filter(like="timer"))

    # -- neat report --
    fields = ['sigma','vid_name','ws','wt']
    fields_summ = ['ws','wt']
    res_fields = ['psnrs','ssims','timer_deno','mem_alloc','mem_res']
    res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f']
    # res_fields = ['psnrs','ssims','timer_deno','timer_extract',
    #               'timer_search','timer_agg','mem_alloc','mem_res']
    # res_fmt = ['%2.3f','%1.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f','%2.3f']

    # -- run agg --
    agg = {key:[np.stack] for key in res_fields}
    grouped = records.groupby(fields).agg(agg)
    grouped.columns = res_fields
    grouped = grouped.reset_index()
    for field in res_fields:
        print(grouped['vid_name'])
        print(field)
        vals = grouped[field].to_numpy()
        for i in range(len(vals)):
            # if i == 18:
            #     print("18")
            #     print(vals[i])
            #     print(np.mean(vals[i][1]))
            print(i,vals[i].shape,type(vals[i][0]))
            print(i,np.mean(vals[i]))
        vals = [np.mean(v) for v in vals]
        print(vals)
        # print(np.concatenate(grouped[field].to_numpy()).shape)
        # vals = list(map(np.mean,grouped[field].to_numpy()))
        grouped[field] = vals#grouped[field].apply(np.mean)
    res_fmt = {k:v for k,v in zip(res_fields,res_fmt)}
    print("\n\n" + "-="*10+"-" + "Report" + "-="*10+"-")
    for sigma,sdf in grouped.groupby("sigma"):
        print("--> sigma: %d <--" % sigma)
        for gfields,gdf in sdf.groupby(fields_summ):
            gfields = list(gfields)

            # -- header --
            header = "-"*5 + " ("
            for i,field in enumerate(fields_summ):
                if field == "pretrained_path":
                    path_str = get_pretrained_path_str(gfields[i])
                    header += "%s, " % (path_str)
                else:
                    header += "%s, " % (gfields[i])
            header = header[:-2]
            header += ") " + "-"*5
            print(header)

            for vid_name,vdf in gdf.groupby("vid_name"):
                # -- res --
                res = "%13s: " % vid_name
                for field in res_fields:
                    res += res_fmt[field] % (vdf[field].mean()) + " "
                print(res)
            # -- res --
            res = "%13s: " % "Ave"
            for field in res_fields:
                res += res_fmt[field] % (gdf[field].mean()) + " "
            print(res)


    # # -- print by dname,sigma --
    # for dname,ddf in records.groupby("dname"):
    #     # field = "internal_adapt_nsteps"
    #     field = "adapt_mtype"
    #     for adapt,adf in ddf.groupby(field):
    #         adapt_psnrs = np.stack(adf['adapt_psnrs'].to_numpy())
    #         print("adapt_psnrs.shape: ",adapt_psnrs.shape)
    #         for cflow,fdf in adf.groupby("flow"):
    #             for ws,wsdf in fdf.groupby("ws"):
    #                 for wt,wtdf in wsdf.groupby("wt"):
    #                     print("adapt,ws,wt,cflow: ",adapt,ws,wt,cflow)
    #                     for sigma,sdf in wtdf.groupby("sigma"):
    #                         ave_psnr,ave_ssim,ave_time,num_vids = 0,0,0,0
    #                         for vname,vdf in sdf.groupby("vid_name"):
    #                             psnr_v = vdf.psnrs[0].mean()
    #                             ssim_v = vdf.ssims[0].mean()
    #                             ave_psnr += psnr_v
    #                             ave_ssim += ssim_v
    #                             ave_time += vdf['timer_deno'].iloc[0]/len(vdf)
    #                             num_vids += 1
    #                             uuid = vdf['uuid'].iloc[0]
    #                             msg = "\t[%s]: %2.2f %s" % (vname,psnr_v,uuid)
    #                             print(msg)

    #                         ave_psnr /= num_vids
    #                         ave_ssim /= num_vids
    #                         ave_time /= num_vids
    #                         total_frames = len(sdf)
    #                         mem = np.stack(vdf['mem_res'].to_numpy()).mean()
    #                         fields = (sigma,ave_psnr,ave_ssim,ave_time,total_frames,mem)
    #                         print("[%d]: %2.3f,%2.3f @ ave %2.2f sec for %d seq at %2.2f" % fields)


if __name__ == "__main__":
    main()

# 29.768 @ ?
# 30.047 @ 1297.05/497
# ...
