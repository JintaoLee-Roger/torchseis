# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import torch
from torch import nn, Tensor
from ...ops.chunked.conv import set_pad_as_zero, Conv3dChunked
from ...ops.chunked.norms import AdaptiveInstanceNorm3d
from ...ops.fuse import ConvInstanceNorm3d, ConvBatchNorm3d


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        norm: nn.Module = nn.InstanceNorm3d,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes // 4,
                               kernel_size=1,
                               bias=False)
        self.bn1 = norm(planes // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes // 4,
                               planes // 4,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = norm(planes // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(planes // 4, planes, kernel_size=1, bias=False)
        self.bn3 = norm(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_insn = isinstance(self.bn1, nn.InstanceNorm3d)
        self.fused = False

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

    def chunk_conv_forward(self, x: Tensor) -> Tensor:
        residual = x

        out = Conv3dChunked(self.conv1, 64)(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = Conv3dChunked(self.conv2, 64)(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = Conv3dChunked(self.conv3, 64)(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)

        return out

    def chunk_forward(self, x: Tensor, padlist: list, t=-1, indices=None):
        res = x

        out = self.conv1(x)
        if self.is_insn and t == 0:
            self.bn1.stats.add_batch(out[indices])
            return out
        elif self.is_insn and not self.bn1.stats.has_statistics():
            self.bn1.stats.compute()
            if self.fused:
                mean = self.bn1.stats.mean
                var = self.bn1.stats.var
                self.conv1 = ConvInstanceNorm3d(self.conv1, self.bn1, mean, var)
                self.bn1.fused = True

        out = self.bn1(out)
        out = self.relu1(out)
        out = set_pad_as_zero(out, padlist)

        out = self.conv2(out)
        if self.is_insn and t == 1:
            self.bn2.stats.add_batch(out[indices])
            return out
        elif self.is_insn and not self.bn2.stats.has_statistics():
            self.bn2.stats.compute()
            if self.fused:
                mean = self.bn2.stats.mean
                var = self.bn2.stats.var
                self.conv2 = ConvInstanceNorm3d(self.conv2, self.bn2, mean, var)
                self.bn2.fused = True

        out = self.bn2(out)
        out = self.relu2(out)
        out = set_pad_as_zero(out, padlist)

        out = self.conv3(out)
        if self.is_insn and t == 2:
            self.bn3.stats.add_batch(out[indices])
            return out
        elif self.is_insn and not self.bn3.stats.has_statistics():
            self.bn3.stats.compute()
            if self.fused:
                mean = self.bn3.stats.mean
                var = self.bn3.stats.var
                self.conv3 = ConvInstanceNorm3d(self.conv3, self.bn3, mean, var)
                self.bn3.fused = True

        out = self.bn3(out)
        out = set_pad_as_zero(out, padlist)

        out += res
        out = self.relu3(out)

        return out

    def replace_norm(self):
        assert isinstance(self.bn1, nn.InstanceNorm3d)
        if isinstance(self.bn1, AdaptiveInstanceNorm3d):
            self.bn1.stats.reset()
            self.bn1.fused = False
            self.bn2.stats.reset()
            self.bn2.fused = False
            self.bn3.stats.reset()
            self.bn3.fused = False
        else:
            self.bn1 = AdaptiveInstanceNorm3d.from_instance_norm(self.bn1)
            self.bn2 = AdaptiveInstanceNorm3d.from_instance_norm(self.bn2)
            self.bn3 = AdaptiveInstanceNorm3d.from_instance_norm(self.bn3)

    def reset(self):
        if isinstance(self.bn1, AdaptiveInstanceNorm3d):
            self.bn1.stats.reset()
            self.bn1.fused = False
            self.bn2.stats.reset()
            self.bn2.fused = False
            self.bn3.stats.reset()
            self.bn3.fused = False

    @torch.no_grad()
    def fuse_model(self):
        if self.fused:
            return
        if not isinstance(self.bn1, nn.BatchNorm3d):
            self.fused = True
            return
        self.conv1 = ConvBatchNorm3d(self.conv1, self.bn1)
        self.bn1 = nn.Identity()
        self.conv2 = ConvBatchNorm3d(self.conv2, self.bn2)
        self.bn2 = nn.Identity()
        self.conv3 = ConvBatchNorm3d(self.conv3, self.bn3)
        self.bn3 = nn.Identity()
        self.fused = True

    @torch.no_grad()
    def unfuse_model(self):
        if not self.fused:
            return
        # if self.is_insn:
        #     self.fused = False
        #     return
        conv1, norm1 = self.conv1.unfuse()
        setattr(self, 'conv1', conv1.eval())
        setattr(self, 'bn1', norm1.eval())
        conv2, norm2 = self.conv2.unfuse()
        setattr(self, 'conv2', conv2.eval())
        setattr(self, 'bn2', norm2.eval())
        conv3, norm3 = self.conv3.unfuse()
        setattr(self, 'conv3', conv3.eval())
        setattr(self, 'bn3', norm3.eval())
        self.fused = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        norm: nn.Module = nn.InstanceNorm3d,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.norm1 = norm(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.norm2 = norm(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.res_conv = None
        if inplanes != planes:
            self.res_conv = nn.Conv3d(inplanes,
                                      planes,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      bias=True)
        self.downsample = downsample
        self.stride = stride
        self.fused = False
        self.is_insn = isinstance(self.norm1, nn.InstanceNorm3d)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)

        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)

        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        out += residual
        out = self.relu2(out)

        return out

    def chunk_conv_forward(self, x: Tensor) -> Tensor:
        residual = x

        out = Conv3dChunked(self.conv1, 128)(x)

        out = self.norm1(out)
        out = self.relu1(out)

        out = Conv3dChunked(self.conv2, 128)(out)

        out = self.norm2(out)

        if self.res_conv is not None:
            residual = Conv3dChunked(self.res_conv, 128)(residual)
        out += residual
        out = self.relu2(out)

        return out

    def chunk_forward(self, x: Tensor, padlist: list, t=-1, indices=None):
        residual = x

        out = self.conv1(x)

        if self.is_insn and t == 0:
            self.norm1.stats.add_batch(out[indices])
            return out
        elif self.is_insn and not self.norm1.stats.has_statistics():
            self.norm1.stats.compute()
            if self.fused:
                mean = self.norm1.stats.mean
                var = self.norm1.stats.var
                self.conv1 = ConvInstanceNorm3d(self.conv1, self.norm1, mean, var)
                self.norm1.fused = True

        out = self.norm1(out)
        out = self.relu1(out)
        out = set_pad_as_zero(out, padlist)

        out = self.conv2(out)

        if self.is_insn and t == 1:
            self.norm2.stats.add_batch(out[indices])
            return out
        elif self.is_insn and not self.norm2.stats.has_statistics():
            self.norm2.stats.compute()
            if self.fused:
                mean = self.norm2.stats.mean
                var = self.norm2.stats.var
                self.conv2 = ConvInstanceNorm3d(self.conv2, self.norm2, mean, var)
                self.norm2.fused = True

        out = self.norm2(out)
        out = set_pad_as_zero(out, padlist)

        if self.res_conv is not None:
            residual = self.res_conv(residual)
            residual = set_pad_as_zero(residual, padlist)
        out += residual
        out = self.relu2(out)

        return out

    def replace_norm(self):
        assert isinstance(self.norm1, nn.InstanceNorm3d)
        if isinstance(self.norm1, AdaptiveInstanceNorm3d):
            self.norm1.stats.reset()
            self.norm1.fused = False
            self.norm2.stats.reset()
            self.norm2.fused = False
        else:
            self.norm1 = AdaptiveInstanceNorm3d.from_instance_norm(self.norm1)
            self.norm2 = AdaptiveInstanceNorm3d.from_instance_norm(self.norm2)

    def reset(self):
        if isinstance(self.norm1, AdaptiveInstanceNorm3d):
            self.norm1.stats.reset()
            self.norm2.stats.reset()
            self.norm1.fused = False
            self.norm2.fused = False

    def fuse_model(self):
        if self.fused:
            return
        if not isinstance(self.norm1, nn.BatchNorm3d):
            self.fused = True
            return
        self.conv1 = ConvBatchNorm3d(self.conv1, self.norm1)
        self.norm1 = nn.Identity()
        self.conv2 = ConvBatchNorm3d(self.conv2, self.norm2)
        self.norm2 = nn.Identity()
        self.fused = True

    def unfuse_model(self):
        if not self.fused:
            return
        # if self.is_insn:
        #     self.fused = False
        #     return
        conv1, norm1 = self.conv1.unfuse()
        setattr(self, 'conv1', conv1.eval())
        setattr(self, 'norm1', norm1.eval())
        conv2, norm2 = self.conv2.unfuse()
        setattr(self, 'conv2', conv2.eval())
        setattr(self, 'norm2', norm2.eval())
        self.fused = False


class StageModule(nn.Module):

    def __init__(
        self,
        stage: int,
        output_branches: int,
        c: int,
        norm: nn.Module = nn.InstanceNorm3d,
    ):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2**i)
            branch = nn.Sequential(
                BasicBlock(w, w, norm=norm),
                BasicBlock(w, w, norm=norm),
                BasicBlock(w, w, norm=norm),
                BasicBlock(w, w, norm=norm),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential(
                    ))  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv3d(c * (2**j),
                                      c * (2**i),
                                      kernel_size=1,
                                      stride=1,
                                      bias=False),
                            norm(c * (2**i)),
                            nn.Upsample(scale_factor=(2.0**(j - i)),
                                        mode='nearest'),
                        ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv3d(c * (2**j),
                                          c * (2**j),
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          bias=False),
                                norm(c * (2**j)),
                                nn.ReLU(inplace=True),
                            ))
                    ops.append(
                        nn.Sequential(
                            nn.Conv3d(c * (2**j),
                                      c * (2**i),
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bias=False),
                            norm(c * (2**i)),
                        ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused