# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU).
# All rights reserved.

"""
ChannelSeg3d is a 3D channel segmentation model for seismic images, which adopts the Deeplab V3 architecture.

To cite this model:
```text
@article{gao2021channelseg3d,
    title={ChannelSeg3D: Channel simulation and deep learning for channel interpretation in 3D seismic images},
    author={Gao, Hang and Wu, Xinming and Liu, Guofeng},
    journal={Geophysics},
    volume={86},
    number={4},
    pages={IM73--IM83},
    year={2021},
    publisher={Society of Exploration Geophysicists}
}
```
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ._channelseg3d_block import ResNet, ASPP, Decoder
from ...ops.chunked.interpolate import interpolate3d_chunked
import time

class ChannelSeg3d(nn.Module):
    def __init__(self, num_classes=2, out_stride=16, norm=nn.BatchNorm3d):
        super(ChannelSeg3d, self).__init__()
        self.num_classes = num_classes
        self.out_stride = out_stride

        self.resnet = ResNet(norm=norm)
        self.aspp = ASPP(norm)
        self.decoder = Decoder(num_classes, norm=norm)


    def forward(self, x: Tensor, rank=0) -> Tensor:
        if rank != 0:
            return self.forward2(x)
        origsize = x.size()[2:]
        x,low_level_feat = self.resnet(x) # [c=2048,H/16], [c=256, H/4]
        x = self.aspp(x) # [c=256, H/16]
        x = self.decoder(x, low_level_feat) # [c=2, H/4]
        x = F.interpolate(x, size=origsize, mode='trilinear', align_corners=True)
        return x

    @torch.no_grad()
    def forward2(self, x: Tensor, *args, **kwargs) -> Tensor:
        assert not self.training
        origsize = x.size()[2:]
        x, low_level_feat = self._resnet_chunk(x)
        x = self.aspp(x)
        x = self.decoder._forward_conv_chunk(x, low_level_feat)
        if origsize[0] * origsize[1] * origsize[2] > 1024**3:
            x = interpolate3d_chunked(x, 32, 4, 'trilinear', True)
        else:
            x = F.interpolate(x, size=origsize, mode='trilinear', align_corners=True)
        return x

    def _resnet_chunk(self, x: Tensor) -> Tensor:
        """
        Chunked forward pass for the ResNet part.
        """
        x, low_level_feat = self.resnet._forward_chunk(x)
        return x, low_level_feat