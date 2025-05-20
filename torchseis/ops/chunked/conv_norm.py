# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

from torch import Tensor, nn
from torch.nn import functional as F
from .statistics import ChunkStatistics


class ConvNorm(nn.Module):
    """
    chunked cumpute a nn.Conv3d followed by a norm
    """

    def __init__(self, conv: nn.Conv3d, norm, act=None) -> None:
        super().__init__()
        self.ele_wise = False
        self.stat = ChunkStatistics(5, norm=norm)
        self.conv = conv
        self.norm = norm
        self.act = act if act is not None else nn.Identity()

    def statistics(self, x):
        assert self.ele_wise

    def forward(self, x):
        pass

    def chunked_forward(self, x):
        assert self.ele_wise or self.stat.has_statistics()
        pass
