# Copyright (c) 2025 Jintao Li.
# Zhejiang University (ZJU).
# University of Science and Technology of China (USTC).
# All rights reserved.

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ChunkStatistics:

    def __init__(
        self,
        ndim,
        reduction_dims=None,
        norm=None,
        algorithm='welford',
    ):
        """
        Parameters:
        ------------
        - reduction_dims
        - algorithm: welford or naive
        """
        if not algorithm in ['naive', 'welford']:
            raise TypeError("Algorithm must be a StatsAlgorithm enum")

        if norm is not None and isinstance(norm, nn.Module):
            if isinstance(norm, nn.InstanceNorm3d):
                norm = 'instance_norm'
            elif isinstance(norm, nn.GroupNorm):
                norm = 'group_norm'
            elif isinstance(norm, nn.LayerNorm):
                norm = 'layer_norm'
            else:
                raise ValueError(f"Unsupported norm: {norm}")

        assert norm is None or norm in [
            'instance_norm', 'group_norm', 'layer_norm'
        ]

        self.algorithm = algorithm
        self.device = 'cpu'
        self.dtype = torch.float32
        self.ndim = ndim

        if reduction_dims is not None:
            self.reduction_dims = tuple(sorted(reduction_dims))
        elif reduction_dims is None and norm is not None:
            if norm == 'instance_norm':
                self.reduction_dims = tuple(range(2, ndim))
            elif norm == 'group_norm':
                self.reduction_dims = tuple(range(2, ndim))
            elif norm == 'layer_norm':
                self.reduction_dims = tuple(range(1, ndim))
        else:
            self.reduction_dims = tuple(range(ndim))

        self._mean = None
        self._var = None

        if self.algorithm == 'welford':
            self._init_welford()
        else:
            self._init_naive()

    def _init_naive(self):
        self.sum_x = None
        self.sum_x2 = None
        self.n_samples = 0

    def _init_welford(self):
        self.n = 0
        self.current_mean = None
        self.current_M2 = None

    def add_batch(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("Input must be torch.Tensor")

        if x.numel() == 0:
            return
        self.device = x.device
        self.dtype = x.dtype

        if self.algorithm == 'welford':
            self._welford_update(x)
        else:
            self._naive_update(x)

    def _naive_update(self, x):
        # if x.dtype == torch.float16:
        #     raise ValueError(
        #         "float16 is not supported for naive algorithm, because out of precision"
        #     )

        x = x.to(torch.float32)

        batch_samples = 1
        for d in self.reduction_dims:
            if d >= x.ndim:
                raise ValueError(f"Dimension {d} out range of {x.ndim}")
            batch_samples *= x.shape[d]

        sum_x_batch = x.sum(dim=self.reduction_dims)
        sum_x2_batch = (x**2).sum(dim=self.reduction_dims)

        if self.sum_x is None:
            self.sum_x = sum_x_batch.detach().clone()
        else:
            self.sum_x += sum_x_batch.to(self.device)

        if self.sum_x2 is None:
            self.sum_x2 = sum_x2_batch.detach().clone()
        else:
            self.sum_x2 += sum_x2_batch.to(self.device)

        self.n_samples += batch_samples

    def _welford_update(self, x):
        # if x.dtype == torch.float16:
        #     raise ValueError(
        #         "float16 is not supported for naive algorithm, because out of precision"
        #     )
        # x = x.to(self.device)
        x = x.to(torch.float32)

        batch_n = 1
        view_shape = list(x.shape)
        for d in self.reduction_dims:
            if d >= x.ndim:
                raise ValueError(f"Dimension {d} out range of {x.ndim}")
            batch_n *= x.shape[d]
            view_shape[d] = 1
        view_shape = tuple(view_shape)

        with torch.no_grad():
            batch_mean = x.mean(dim=self.reduction_dims)
            centered = x - batch_mean.view(view_shape)
            batch_M2 = (centered**2).sum(dim=self.reduction_dims)

        if self.n == 0:
            self._welford_initialize(batch_mean, batch_M2, batch_n)
        else:
            self._welford_merge(batch_mean, batch_M2, batch_n)

    def _welford_initialize(self, batch_mean, batch_M2, batch_n):
        self.n = batch_n
        self.current_mean = batch_mean.detach().clone()
        self.current_M2 = batch_M2.detach().clone()

    def _welford_merge(self, batch_mean, batch_M2, batch_n):
        total_n = self.n + batch_n
        delta = batch_mean - self.current_mean

        new_mean = self.current_mean + delta * (batch_n / total_n)

        term3 = delta**2 * (self.n * batch_n / total_n)
        new_M2 = self.current_M2 + batch_M2 + term3
        self.current_M2 = new_M2

        self.current_mean = new_mean
        self.n = total_n

    def compute(self):
        if self.algorithm == 'naive' and self.n_samples == 0:
            raise RuntimeError("No batch added")
        if self.algorithm == 'welford' and self.n == 0:
            raise RuntimeError("No batch added")

        if self.algorithm == 'naive':
            self._mean = self.sum_x / self.n_samples
            mean = self._mean if self._mean is not None else self.sum_x / self.n_samples
            self._var = (self.sum_x2 / self.n_samples) - (mean**2)
        else:
            self._mean = self.current_mean.type(torch.float32)
            self._var = self.current_M2 / self.n
            self._var = self._var.type(torch.float32)

        self._mean = self._mean.to(self.dtype)
        self._var = self._var.to(self.dtype)

    @property
    def mean(self):
        if self._mean is None:
            raise RuntimeError("`mean` is not calculated")
        return self._mean

    @property
    def var(self):
        if self._var is None:
            raise RuntimeError("`var` is not calculated")
        return self._var

    def has_statistics(self):
        return self._mean is not None and self._var is not None

    def reset(self):
        self._mean = None
        self._var = None
        if self.algorithm == 'naive':
            self.sum_x = None
            self.sum_x2 = None
            self.n_samples = 0
        else:
            self.n = 0
            self.current_mean = None
            self.current_M2 = None



class ChunkedAdaptiveAvgPool3d1(nn.Module):
    def __init__(self):
        self.acc = None             # accumulator for sum over voxels, shape (N, C)
        self.total_voxels = 0       # total count of voxels added
        self.out = None
        self.device = None
        self.dtype = None
    
    def add_batch(self, x: Tensor):
        """
        Add a chunk of the volume for pooling.

        Args:
            x: Tensor of shape (N, C, d, h, w)
        """
        if not isinstance(x, Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor, got {x.dim()}D")

        N, C, d, h, w = x.shape
        # sum over spatial dims d,h,w -> shape (N, C)
        sum_block = x.float().sum(dim=(2, 3, 4))

        # initialize accumulator
        if self.acc is None:
            self.acc = sum_block.clone()
            self.device = x.device
            self.dtype = x.dtype
        else:
            if sum_block.shape != self.acc.shape:
                raise ValueError(f"Chunk shape mismatch: {sum_block.shape} vs {self.acc.shape}")
            self.acc += sum_block.to(self.device)

        # print(self.acc)
        # update voxel count
        self.total_voxels += d * h * w

    def compute(self) -> Tensor:
        """
        Compute the global average pooling result.

        Returns:
            Tensor of shape (N, C, 1, 1, 1)
        """
        if self.acc is None or self.total_voxels == 0:
            raise RuntimeError("No chunks added for pooling")

        # compute mean over all added voxels
        mean = self.acc / self.total_voxels  # shape (N, C)
        # reshape to (N, C, 1, 1, 1) and cast back
        self.out = mean.to(self.dtype).to(self.device).view(*mean.shape, 1, 1, 1)

    @property
    def output(self) -> Tensor:
        """
        Get the computed output tensor.

        Returns:
            Tensor of shape (N, C, 1, 1, 1)
        """
        if self.out is None:
            raise RuntimeError("Output not computed yet. Call compute() first.")
        return self.out
    
    def has_statistics(self) -> bool:
        return self.out is not None
    
    def reset(self):
        """
        Reset the accumulator and output.
        """
        self.acc = None
        self.out = None
        self.total_voxels = 0
        self.device = None
        self.dtype = None


if __name__ == "__main__":

    import time
    dims = (2, 3)
    x = torch.rand(11, 2, 512, 512).cuda() + torch.randn(11, 2, 512, 512).cuda()
    x = x * 1000
    x = x.half()
    t1 = time.time()
    mm = x.mean(dims)
    vv = x.var(dims)
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    stats = ChunkStatistics(4, reduction_dims=dims)

    for i in range(8):
        for j in range(8):
            stats.add_batch(x[:, :, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64])

    out1 = stats.compute()
    mm1 = stats.mean
    vv1 = stats.var
    t2 = time.time()
    print(t2 - t1)

    t1 = time.time()
    stats2 = ChunkStatistics(4, reduction_dims=dims, algorithm='naive')
    for i in range(8):
        for j in range(8):
            stats2.add_batch(x[:, :, i * 64:(i + 1) * 64, j * 64:(j + 1) * 64])

    out2 = stats2.compute()
    mm2 = stats2.mean
    vv2 = stats2.var
    t2 = time.time()
    print(t2 - t1)

    print(torch.allclose(mm, mm1, atol=1e-6))
    print(torch.allclose(vv, vv1, atol=1e-6))
    print(torch.allclose(mm, mm2, atol=1e-6))
    print(torch.allclose(vv, vv2, atol=1e-6))
