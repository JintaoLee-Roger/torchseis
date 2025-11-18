# Copyright (c) 2025 Jintao Li.
# Zhejiang University (ZJU).
# University of Science and Technology of China (USTC).
# All rights reserved.
"""
This file contains some manual implementations of normalization functions.

These functions are used to chunked calculation by feeding the pre-computed `mean` and `var`.
"""

import itertools
import torch
from torch import Tensor, nn
from .statistics import ChunkStatistics


def cal_mean_and_var(x: Tensor, norm='instance'):
    stats = ChunkStatistics(5, norm=norm)
    b, c, d, h, w = x.shape
    nd, nh, nw = 1, 1, 1
    if d % 2 == 0: nd = 2
    if h % 2 == 0: nh = 2
    if w % 2 == 0: nw = 2
    bd, bh, bw = d//nd, h//nh, w//nw

    for ic in range(c):
        for i, j, k in itertools.product(range(nd), range(nh), range(nw)):
            stats.add_batch(x[:, ic:ic+1, i*bd:(i+1)*bd, j*bh:(j+1)*bh, k*bw:(k+1)*bw])
    stats.compute()
    return stats.mean, stats.var


class AdaptiveInstanceNorm3d(nn.InstanceNorm3d):

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(num_features, eps, momentum, affine, *args, **kwargs)
        with torch.no_grad():
            if weight is not None:
                self.weight = weight
            if bias is not None:
                self.bias = bias
        self.stats = ChunkStatistics(5, norm='instance_norm')
        self.fused = False

    def forward(
        self,
        x: Tensor,
        mean: Tensor = None,
        var: Tensor = None,
        **kwargs,
    ) -> Tensor:
        if self.fused:
            return x
        if (mean is None or var is None) and not self.stats.has_statistics():
            return super().forward(x, **kwargs)
        else:
            if mean is None or var is None:
                mean = self.stats.mean
                var = self.stats.var
            assert torch.squeeze(mean).shape == torch.squeeze(var).shape
            return instance_norm(x, mean, var, self.eps, self.affine,
                                 self.weight, self.bias, True)

    @classmethod
    def from_instance_norm(
        cls,
        instance_norm: nn.InstanceNorm3d,
    ) -> "AdaptiveInstanceNorm3d":
        assert isinstance(instance_norm, nn.InstanceNorm3d)
        return cls(
            instance_norm.weight,
            instance_norm.bias,
            instance_norm.num_features,
            instance_norm.eps,
            instance_norm.momentum,
            instance_norm.affine,
        )


class AdaptiveGroupNorm(nn.GroupNorm):

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        weight: Tensor = None,
        bias: Tensor = None,
        *args,
        **kwargs,
    ):
        super().__init__(num_groups, num_channels, eps, affine, *args, **kwargs) # yapf: disable
        with torch.no_grad():
            if weight is not None:
                self.weight = weight
            if bias is not None:
                self.bias = bias
        self.stats = ChunkStatistics(5, norm='group_norm')

    def forward(
        self,
        x: Tensor,
        mean: Tensor = None,
        var: Tensor = None,
        inplace: bool = True,
        **kwargs,
    ) -> Tensor:
        if (mean is None or var is None) and not self.stats.has_statistics() and not inplace:
            return super().forward(x, **kwargs)
        else:
            if mean is None or var is None:
                if self.stats.has_statistics():
                    mean = self.stats.mean
                    var = self.stats.var
            # assert torch.squeeze(mean).shape == torch.squeeze(var).shape
            return group_norm(x, mean, var, self.num_groups, self.eps,
                              self.affine, self.weight, self.bias, True)

    @classmethod
    def from_group_norm(
        cls,
        group_norm: nn.GroupNorm,
    ) -> "AdaptiveGroupNorm":
        assert isinstance(group_norm, nn.GroupNorm)
        return cls(
            group_norm.num_groups,
            group_norm.num_channels,
            group_norm.eps,
            group_norm.affine,
            group_norm.weight,
            group_norm.bias,
        )


def instance_norm(
    x: Tensor,
    mean: Tensor = None,
    var: Tensor = None,
    eps: float = 1e-5,
    affine: bool = True,
    weight: Tensor = None,
    bias: Tensor = None,
    inplace: bool = False,
    keep_dtype: bool = False,
) -> Tensor:
    """
    Instance Normalization for multi-dimensional inputs (N, C, *), supporting float16 computation.
    
    Args:
        x (Tensor): Input tensor of shape (N, C, *).
        mean (Tensor, optional): Precomputed mean. If None, computed from x.
        var (Tensor, optional): Precomputed variance. If None, computed from x.
        eps (float): Small value for numerical stability.
        affine (bool): If True, apply learnable affine transformation.
        weight (Tensor, optional): Scale parameter (gamma) if affine=True.
        bias (Tensor, optional): Shift parameter (beta) if affine=True.
        inplace (bool): If True, modify x in-place.
    
    Returns:
        Tensor: Normalized output tensor.
    """
    input_shape = x.shape
    input_dtype = x.dtype
    N, C = input_shape[:2]
    x = x.view(N, C, -1)
    spatial_dims = tuple(range(2, x.dim()))

    if input_dtype == torch.float16 and not keep_dtype:
        x = x.float()
        if mean is not None: mean = mean.float()
        if var is not None: var = var.float()
        if weight is not None: weight = weight.float()
        if bias is not None: bias = bias.float()

    if mean is None:
        mean = x.mean(dim=spatial_dims, keepdim=True)
    if var is None:
        var = x.var(dim=spatial_dims, keepdim=True, unbiased=False)

    mean = mean.view(N, C, 1)
    var = var.view(N, C, 1)

    if not inplace:
        x = (x - mean) / torch.sqrt(var + eps)
    else:
        x.sub_(mean).div_(torch.sqrt(var + eps))

    if affine and weight is not None and bias is not None:
        weight = weight.view(1, C, 1)
        bias = bias.view(1, C, 1)
        if not inplace:
            x = x * weight + bias
        else:
            x.mul_(weight).add_(bias)

    if input_dtype == torch.float16:
        x = x.to(input_dtype)

    return x.view(input_shape)


def group_norm(
    x: Tensor,
    mean: Tensor = None,
    var: Tensor = None,
    num_groups: int = 32,
    eps: float = 1e-5,
    affine: bool = True,
    weight: Tensor = None,
    bias: Tensor = None,
    inplace: bool = False,
) -> Tensor:
    N, C, D, H, W = x.shape
    assert C % num_groups == 0, "The number of channels must be divisible by the number of groups"
    x = x.reshape(N, num_groups, C // num_groups, D, H, W)
    x = x.reshape(N, num_groups, -1)
    if mean is None:
        mean = x.mean(dim=2, keepdim=True)
    if var is None:
        var = x.var(dim=2, keepdim=True, unbiased=False)
    mean = mean.reshape(N, num_groups, 1)
    var = var.reshape(N, num_groups, 1)
    if not inplace:
        x = (x - mean) / torch.sqrt(var + eps)
    else:
        x.sub_(mean).div_(torch.sqrt(var + eps))

    if affine and weight is not None and bias is not None:
        weight = weight.reshape(1, C, 1)
        bias = bias.reshape(1, C, 1)
        if not inplace:
            x = x * weight + bias
        else:
            x.mul_(weight).add_(bias)

    x = x.reshape(N, C, D, H, W)

    return x
