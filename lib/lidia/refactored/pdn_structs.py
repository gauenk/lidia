

# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- neural network --
import torch.nn as nn

# -- submodules --
from .sep_structs import SeparableFcNet,FcNet

from .misc import print_gpu_stats

class PatchDenoiseNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size, gpu_stats):
        super(PatchDenoiseNet, self).__init__()

        # -- options --
        self.arch_opt = arch_opt
        self.gpu_stats = gpu_stats

        # -- sep filters --
        self.sep_net = SeparableFcNet(arch_opt=arch_opt,
                                      patch_w=patch_w,
                                      ver_size=ver_size)
        # -- sep 0 --
        self.weights_net0 = FcNet()
        self.alpha0 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        # -- sep 1 --
        self.weights_net1 = FcNet()
        self.alpha1 = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

        # -- combo --
        self.beta = nn.Parameter(th.tensor((0.5,), dtype=th.float32))

    def forward(self,patches_n0,dist0,inds0,params0,patches_n1,dist1,inds1,params1):
        """
        Run patch denoiser network
        """

        #
        # -- Sep @ 0 --
        #

        weights0 = self.weights_net0(th.exp(-self.alpha0.abs() * dist0))
        weights0 = weights0.unsqueeze(-1)
        wpatches0 = patches_n0 * weights0
        h,w = params0['pixels_h'],params0['pixels_w']
        agg0,sep0,_ = self.sep_net.run_sep0(wpatches0,dist0,inds0,h,w)
        self.print_gpu_stats("Sep0")

        #
        # -- Sep @ 1 --
        #

        weights1 = self.weights_net1(th.exp(-self.alpha1.abs() * dist1))
        weights1 = weights1.unsqueeze(-1)
        wpatches1 = patches_n1 * weights1
        weights1 = weights1[:, :, 0:1, :]
        h,w = params1['pixels_h'],params1['pixels_w']
        agg1,sep1,_ = self.sep_net.run_sep1(wpatches1,weights1,dist1,inds1,h,w)
        assert th.any(th.isnan(agg1)).item() is False
        self.print_gpu_stats("Sep1")

        #
        # -- Final Sep --
        #

        inputs = th.cat((sep0, sep1, agg0, agg1), dim=-2)
        noise = self.sep_net.sep_part2(inputs)
        deno,patches_w = self.run_pdn_final(patches_n0,noise)
        self.print_gpu_stats("Final")

        return deno,patches_w

    def run_pdn_final(self,patches_n0,noise):
        patches_dn = patches_n0[:, :, 0, :] - noise.squeeze(-2)
        patches_no_mean = patches_dn - patches_dn.mean(dim=-1, keepdim=True)
        patch_exp_weights = (patches_no_mean ** 2).mean(dim=-1, keepdim=True)
        patch_weights = th.exp(-self.beta.abs() * patch_exp_weights)
        return patches_dn,patch_weights

    def print_gpu_stats(self,name="-"):
        print_gpu_stats(self.gpu_stats,name)
