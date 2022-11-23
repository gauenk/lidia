"""
Wrap the opencv optical flow

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- estimate sigma --
from skimage.restoration import estimate_sigma

# -- misc --
from easydict import EasyDict as edict

# -- opencv --
import cv2 as cv

# -- local --
from ..utils import color

def run_batch(vid,sigma):
    B = vid.shape[0]
    flows = edict()
    flows.fflow,flows.bflow = [],[]
    for b in range(B):
        flows_b = run(vid[b],sigma)
        flows.fflow.append(flows_b.fflow)
        flows.bflow.append(flows_b.bflow)
    flows.fflow = th.stack(flows.fflow)
    flows.bflow = th.stack(flows.bflow)
    return flows

def run(vid_in,sigma):

    # -- init --
    device = vid_in.device
    vid_in = vid_in.cpu()
    vid = vid_in.clone() # copy data for no-rounding-error from RGB <-> YUV
    t,c,h,w = vid.shape

    # -- color2gray --
    vid = th.clamp(vid,0,255.).type(th.uint8)
    if vid.shape[1] == 3:
        color.rgb2yuv(vid)
    vid = vid[:,[0],:,:]
    vid = rearrange(vid,'t c h w -> t h w c')

    # -- alloc --
    fflow = th.zeros((t,2,h,w),device=device)
    bflow = th.zeros((t,2,h,w),device=device)

    # -- computing --
    for ti in range(t-1):
        fflow[ti] = pair2flow(vid[ti],vid[ti+1],device)
    for ti in reversed(range(t-1)):
        bflow[ti+1] = pair2flow(vid[ti+1],vid[ti],device)

    # -- final shaping --
    # fflow = rearrange(fflow,'t h w c -> t c h w')
    # bflow = rearrange(bflow,'t h w c -> t c h w')

    # -- packing --
    flows = edict()
    flows.fflow = fflow
    flows.bflow = bflow

    # -- gray2color --
    color.yuv2rgb(vid)

    return flows

def est_sigma(vid):
    if vid.shape[1] == 3:
        vid = vid.cpu().clone()
        color.rgb2yuv(vid)
    vid_np = vid.cpu().numpy()
    vid_np = vid_np[:,[0]] # Y only
    sigma = estimate_sigma(vid_np,channel_axis=1)[0]
    return sigma

def pair2flow(frame_a,frame_b,device):
    if "cpu" in str(frame_a.device):
        return pair2flow_cpu(frame_a,frame_b,device)
    else:
        return pair2flow_gpu(frame_a,frame_b,device)

def pair2flow_cpu(frame_a,frame_b,device):

    # -- numpy --
    frame_a = frame_a.cpu().numpy()
    frame_b = frame_b.cpu().numpy()

    # -- exec flow --
    flow = cv.calcOpticalFlowFarneback(frame_a,frame_b,
                                        0.,0.,3,15,3,5,1.,0)
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow

def pair2flow_gpu(frame_a,frame_b,device):

    # -- create opencv-gpu frames --
    gpu_frame_a = cv.cuda_GpuMat()
    gpu_frame_b = cv.cuda_GpuMat()
    gpu_frame_a.upload(frame_a.cpu().numpy())
    gpu_frame_b.upload(frame_b.cpu().numpy())

    # -- create flow object --
    gpu_flow = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False,
                                                   15, 3, 5, 1.2, 0)

    # -- exec flow --
    flow = cv.cuda_FarnebackOpticalFlow.calc(gpu_flow, gpu_frame_a,
                                             gpu_frame_b, None)
    flow = flow.download()
    flow = flow.transpose(2,0,1)
    flow = th.from_numpy(flow).to(device)

    return flow
