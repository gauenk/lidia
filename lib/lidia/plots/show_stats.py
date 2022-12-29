
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- pandas --
import pandas as pd

# -- plotting --
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# -- save --
from pathlib import Path
SAVE_DIR = Path("./output/plots/show_stats")

def create_stat_plots(records):

    # -- mkdir --
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)

    # -- split successful records --
    df = records[records["failed"] ==0]
    ldf = df[df['mtype']=="original"]
    bdf = df[df['mtype']=="batched"]

    # -- get max non-failing batch size --
    print(bdf)
    bdf_l = []
    for res,rdf in bdf.groupby("res"):
        max_batch_perc = rdf['batch_perc'].max()
        bdf_m = rdf[rdf['batch_perc'] == max_batch_perc]
        bdf_l.append(bdf_m)
    bdf = pd.concat(bdf_l)
    print(bdf)

    # -- order fields --
    fields = ["res",'batch_perc','timer_deno']
    ldf = ldf.sort_values("res")
    bdf = bdf.sort_values("res")
    print(ldf[fields])
    print(bdf[fields])


    # -- create plots --
    plot_res_v_bs(ldf,bdf)
    plot_res_v_runtime(ldf,bdf)


def plot_res_v_bs(ldf,bdf):

    # -- params --
    FSIZE = 14
    FSIZE_S = 12

    # -- get lidia data --
    ldf_res = np.log10(ldf['res']**2)
    ldf_bs = np.log10(ldf['res']**2)

    # -- get n4net data --
    bdf_res = np.log10(bdf['res']**2)
    bdf_bs = np.log10(bdf['batch_perc'] * bdf['res']**2)

    # -- create plots --
    fig,ax = plt.subplots(figsize=(4,3))

    # -- plot --
    ax.plot(bdf_res,bdf_bs,'-x',label="ours")
    ax.plot(ldf_res,ldf_bs,'-+',label="lidia")

    # -- label axis --
    ax.set_title("Our Method Scales by Batching Pixels",fontsize=FSIZE)
    ax.set_ylabel("Batch Size [log10]",fontsize=FSIZE)
    ax.set_xlabel("Image Resolution [log10]",fontsize=FSIZE)
    xticks = np.linspace(bdf_res.min(),bdf_res.max(),4)
    yticks = np.linspace(bdf_bs.min(),bdf_bs.max(),4)
    xticklabels = ["%1.1f" % x for x in xticks]
    yticklabels = ["%1.1f" % y for y in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE_S)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE_S)

    # -- mkdir --
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)

    # -- save --
    ax.legend(fontsize=FSIZE_S,title_fontsize=FSIZE,title="Method")
    fname = SAVE_DIR / "show_stats_res_v_bs.png"
    plt.savefig(fname,bbox_inches="tight",dpi=300)

    # -- close --
    plt.cla()
    plt.clf()
    plt.close("all")

def plot_res_v_runtime(ldf,bdf):

    # -- params --
    FSIZE = 14
    FSIZE_S = 12

    # -- get lidia data --
    ldf_res = np.log10(ldf['res']**2)
    ldf_run = ldf['timer_deno']

    # -- get n4net data --
    bdf_res = np.log10(bdf['res']**2)
    bdf_run = bdf['timer_deno']

    # -- create plots --
    fig,ax = plt.subplots(figsize=(4,3))

    # -- plot --
    ax.plot(bdf_res,bdf_run,'-x',label="ours")
    ax.plot(ldf_res,ldf_run,'-+',label="lidia")

    # -- a slope=1 line for scale --
    print(bdf_run[-4:-2])
    x_pts = np.copy(bdf_run[-4:-2].to_numpy())
    x_pts = x_pts - x_pts.min() + 5.7#bdf_res[-5:-4].mean()
    print(x_pts)
    print(bdf_run[-4:-2])
    ax.plot(x_pts,bdf_run[-4:-2],'-.',label="slope=1")

    # -- label axis --
    ax.set_title("Runtime Scales Almost Linearly with Resolution",fontsize=FSIZE)
    ax.set_ylabel("Runtime (sec)",fontsize=FSIZE)
    ax.set_xlabel("Image Resolution [log10]",fontsize=FSIZE)
    xticks = np.linspace(bdf_res.min(),bdf_res.max(),4)
    yticks = np.linspace(bdf_run.min(),bdf_run.max(),4)
    xticklabels = ["%1.1f" % x for x in xticks]
    yticklabels = ["%1.2f" % y for y in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=FSIZE_S)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=FSIZE_S)

    # -- mkdir --
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True)

    # -- save --
    ax.legend(fontsize=FSIZE_S,title_fontsize=FSIZE,title="Method")
    fname = SAVE_DIR / "show_stats_res_v_runtime.png"
    plt.savefig(fname,bbox_inches="tight",dpi=300)

    # -- close --
    plt.cla()
    plt.clf()
    plt.close("all")

