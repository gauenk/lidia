
# -- linalg --
import torch as th
from einops import rearrange

# -- neural network --
import torch.nn as nn

# -- submodules --
from .bn_structs import VerHorBnRe,VerHorMat
from .agg_structs import Aggregation0,Aggregation1

class FcNet(nn.Module):
    def __init__(self,name):
        super(FcNet, self).__init__()
        self.name = name
        for layer in range(6):
            self.add_module('fc{}'.format(layer), nn.Linear(in_features=14,
                                                            out_features=14,
                                                            bias=False))
            self.add_module('bn{}'.format(layer), nn.BatchNorm1d(14))
            self.add_module('relu{}'.format(layer), nn.ReLU())

        self.add_module('fc_out', nn.Linear(in_features=14, out_features=14, bias=True))

    def forward(self, x):
        for name, module in self._modules.items():
            if 'bn' in name:
                images, patches, values = x.shape
                x = module(x.view(images * patches, values)).view(images, patches, values)
            else:
                x = module(x)
        return x

class SeparablePart1(nn.Module):
    def __init__(self, arch_opt, hor_size, patch_numel, ver_size,name=""):
        super(SeparablePart1, self).__init__()

        self.ver_hor_bn_re0 = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size,
                                         hor_in=14, hor_out=hor_size, bn=False)
        self.a = nn.Parameter(th.tensor((0,), dtype=th.float32))
        self.ver_hor_bn_re1 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size,
                                         hor_in=14, hor_out=hor_size, bn=True,
                                         name=name)

    def forward(self, x):
        x = self.ver_hor_bn_re0(x)
        if hasattr(self, 'ver_hor_bn_re1'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re1(x)
        return x

class SeparablePart2(nn.Module):
    def __init__(self, arch_opt, hor_size_in, patch_numel, ver_size):
        super(SeparablePart2, self).__init__()
        self.ver_hor_bn_re2 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size,
                                         hor_in=hor_size_in, hor_out=56, bn=True,
                                         name="sep2_a")
        self.a = nn.Parameter(th.tensor((0,), dtype=th.float32))
        self.ver_hor_bn_re3 = VerHorBnRe(ver_in=ver_size, ver_out=ver_size,
                                         hor_in=56, hor_out=56, bn=True,
                                         name="sep2_b")
        self.ver_hor_out = VerHorMat(ver_in=ver_size, ver_out=patch_numel,
                                     hor_in=56, hor_out=1)

    def forward(self, x):
        x = self.ver_hor_bn_re2(x)
        if hasattr(self, 'ver_hor_bn_re3'):
            x = self.a * x + (1 - self.a) * self.ver_hor_bn_re3(x)
        x = self.ver_hor_out(x)

        return x


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       All Steps Together!
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class SeparableFcNet(nn.Module):
    def __init__(self, arch_opt, patch_w, ver_size, grad_sep_part1):
        super(SeparableFcNet, self).__init__()
        patch_numel = (patch_w ** 2) * 3 if arch_opt.rgb else patch_w ** 2

        # -- grad parameters --
        self.grad_sep_part1 = grad_sep_part1

        # -- sep nets [0 & 1] --
        self.sep_part1_s0 = SeparablePart1(arch_opt=arch_opt, hor_size=14,
                                           patch_numel=patch_numel, ver_size=ver_size,
                                           name="sep1_a")
        self.sep_part1_s1 = SeparablePart1(arch_opt=arch_opt, hor_size=14,
                                           patch_numel=patch_numel, ver_size=ver_size,
                                           name="sep1_b")

        # -- sep 0 --
        self.agg0 = Aggregation0(patch_w)
        self.agg0_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel,
                                  hor_in=14, hor_out=1)
        self.agg0_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size,
                                    hor_in=1, hor_out=14, bn=False)

        # -- sep 1 --
        self.agg1 = Aggregation1(patch_w)
        self.agg1_pre = VerHorMat(ver_in=ver_size, ver_out=patch_numel,
                                  hor_in=14, hor_out=1)
        self.agg1_post = VerHorBnRe(ver_in=patch_numel, ver_out=ver_size,
                                    hor_in=1, hor_out=14, bn=False)

        # -- combo seps --
        self.sep_part2 = SeparablePart2(arch_opt=arch_opt, hor_size_in=56,
                                        patch_numel=patch_numel, ver_size=ver_size)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #           Split Sep Steps
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    #
    # -- Sep 0 --
    #

    def run_batched_sep0_a(self,wpatches,weights,qindex,pfxns,wdiv=False):
        x_out = self.sep_part1_s0(wpatches)
        y_out = self.agg0_pre(x_out)
        if wdiv: y_out /= weights
        vid = self.agg0.batched_fwd_a(y_out, qindex, pfxns.fold, pfxns.wfold)
        return vid,x_out

    def run_batched_sep0_b(self,wpatches,weights,vid,qindex,bsize,unfold,wdiv=False):
        x_out = self.sep_part1_s0(wpatches)
        if self.grad_sep_part1: x_out = x_out.detach()
        y_out = self.agg0.batched_fwd_b(vid,qindex,bsize,unfold)
        y_out = self.agg0_post(y_out)
        return y_out,x_out

    #
    # -- Sep 1 --
    #

    def run_batched_sep1_a(self,wpatches,weights,qindex,pfxns,wdiv=True):
        x_out = self.sep_part1_s1(wpatches)
        y_out = self.agg1_pre(x_out)
        if wdiv: y_out /= weights
        vid = self.agg1.batched_fwd_a(y_out, qindex, pfxns.fold, pfxns.wfold)
        return vid,x_out

    def run_batched_sep1_b(self,wpatches,weights,vid,qindex,bsize,unfold,wdiv=True):
        x_out = self.sep_part1_s1(wpatches)
        if self.grad_sep_part1: x_out = x_out.detach()
        y_out = self.agg1.batched_fwd_b(vid,qindex,bsize,unfold)
        if wdiv: y_out = weights * y_out
        y_out = self.agg1_post(y_out)
        return y_out,x_out

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #           Full Sep Steps
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def run_sep0(self,wpatches,dists,inds,h,w):
        x = self.sep_part1_s0(wpatches)
        y_out = self.agg0_pre(x)
        y_out,fold_out = self.agg0(y_out, dists, inds, h, w, both=True)
        y_out = self.agg0_post(y_out)
        return y_out,x,fold_out

    def run_sep1(self,wpatches,weights,dists,inds,h,w):
        x = self.sep_part1_s1(wpatches)
        y_out = self.agg1_pre(x) / weights
        y_out,fold_out = self.agg1(y_out, dists, inds, h, w, both=True)
        y_out = self.agg1_post(weights * y_out)
        return y_out,x,fold_out

    def forward(self):
        raise NotImplemented("")

