# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU).
# All rights reserved.

"""
FaultSSL model is a semisupervised learning model for seismic fault detection.

To cite this model:
```text
@article{dou2024faultssl,
  title={FaultSSL: Seismic fault detection via semisupervised learning},
  author={Dou, Yimin and Li, Kewen and Dong, Minghui and Xiao, Yuan},
  journal={Geophysics},
  volume={89},
  number={3},
  pages={M79--M91},
  year={2024},
  publisher={Society of Exploration Geophysicists},
  doi={10.1190/geo2023-0550.1},
  url={https://library.seg.org/doi/10.1190/geo2023-0550.1}
}
```
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from torch.ao.quantization import fuse_modules
from ._faultssl_modules import *
from ._faultssl_chunk import FaultSSLChunk
from ...ops.chunked.interpolate import interpolate3d_chunked
from ...ops.chunked.conv import ConvTranspose3dChunked
from ...ops.fuse import ConvInstanceNorm3d, ConvBatchNorm3d

__all__ = ['FaultSSL']


class FaultSSL(nn.Module, FaultSSLChunk):
    """
    Implements FaultSSL model from
    `"FaultSSL: Seismic fault detection via semisupervised learning"
    <https://library.seg.org/doi/abs/10.1190/geo2023-0550.1>`_.

    Cost tested on a H20, (precision/precision-fuse/iou)
    ---------------------
    | rank | shape              | memory   | time   | GPU        |
    |------|--------------------|----------|--------|------------|
    | 0    | (512, 512, 512)    | 57.93 GB | 72.60s | H20 96 GB  |
    | 0    | (768, 512, 512)    | 84.06 GB | 120.7s | H20 96 GB  |
    |------|--------------------|----------|--------|------------|
    | 1    | (768, 768, 704)    | 23.39/23.39/24.04 GB | 20.88/13.66/62.32s | RTX 24 GB  |
    | 1    | (832, 832, 832)    | 31.78/31.78/32.42 GB | 29.03/18.98/86.24s | V100 32 GB |
    | 1    | (1024, 832, 832)   | 38.70/38.70/39.35 GB | 35.50/22.97/105.76s | A100 40 GB |
    | 1    | (1024, 1024, 1024) | 57.73/57.73/58.19 GB | 53.43/34.42/162.04s | A100 80 GB |
    | 1    | (1280, 1152, 1024) | 80.48/80.48/80.94 GB | 74.90/48.18/226.94s | A100 80 GB |
    | 1    | (1280, 1280, 1152) | 90.19/90.19/90.18 GB | 93.91/60.57/287.51s | H20 96 GB  |

    Note
    -------
    HRNet will produce large amounts of broken GPU memory fragments. If feed a large tensor, 
    these fragments will be recycled by pytorch and take a lot of time when the unallocated memory is not sufficient.
    This is obvious when rise the tensor size from (768, 672, 672) to (1024, 768, 768).

    Parameters
    ------------
    base : int
        The base number of channels for the first layer. Default is 24.
    mode : str
        The mode to use for the model, choices from ['iou', 'precision']. Default is 'iou'.
    """

    def __init__(self, base: int = 24, mode: str = 'iou'):
        super(FaultSSL, self).__init__()

        self.need_padding = 2**4
        self.name = 'FaultSSL'

        self.fused = False
        self.base = base
        c = base
        assert mode in ['iou', 'precision']
        self.mode = mode
        if mode == 'iou':
            norm = lambda channels: nn.InstanceNorm3d(channels, affine=True)
        elif mode == 'precision':
            norm = nn.BatchNorm3d
        else:
            raise ValueError(
                f"Invalid mode: {mode}, must be 'iou' or 'precision'")

        # self.prepad = torch.nn.ReflectionPad3d([32, 32, 32, 32, 0, 0])
        self.input_layer1 = nn.Sequential(
            nn.Conv3d(1, c, 3, stride=1, padding=1, bias=False),
            norm(c),
            nn.ReLU(inplace=True),
            Bottleneck(c, c, norm=norm),
            Bottleneck(c, c, norm=norm),
        )

        self.input_layer2 = nn.Sequential(
            nn.Conv3d(c, c * 2, 3, stride=2, padding=1, bias=False),
            norm(c * 2),
            nn.ReLU(inplace=True),
            Bottleneck(c * 2, c * 2, norm=norm),
            Bottleneck(c * 2, c * 2, norm=norm),
        )

        self.input_layer3 = nn.Sequential(
            nn.Conv3d(c * 2, c * 8, 3, stride=2, padding=1, bias=False),
            norm(c * 8),
            nn.ReLU(inplace=True),
            Bottleneck(c * 8, c * 8, norm=norm),
            Bottleneck(c * 8, c * 8, norm=norm),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(c * 8, c * 2, 3, stride=1, padding=1, bias=False),
                norm(c * 2),
                nn.ReLU(inplace=True),
            ),
            # Double Sequential to fit with official pretrained weights
            nn.Sequential(
                nn.Sequential(
                    nn.Conv3d(c * 8, c * 4, 3, stride=2, padding=1,
                              bias=False),
                    norm(c * 4),
                    nn.ReLU(inplace=True),
                )),
        ])

        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=int(c * 2), norm=norm),
            StageModule(stage=2, output_branches=2, c=int(c * 2), norm=norm),
            StageModule(stage=2, output_branches=2, c=int(c * 2), norm=norm),
        )

        self.transition2 = nn.ModuleList([
            # None,   - Used in place of "None" because it is callable
            nn.Sequential(),
            nn.Sequential(),
            # Double Sequential to fit with official pretrained weights
            nn.Sequential(
                nn.Sequential(
                    nn.Conv3d(
                        int(c * 2) * (2**1),
                        int(c * 2) * (2**2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    norm(int(c * 2) * (2**2)),
                    nn.ReLU(inplace=True),
                )),
        ])

        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
            StageModule(stage=3, output_branches=3, c=int(c * 2), norm=norm),
        )

        self.Decoder1 = nn.Sequential(
            nn.ConvTranspose3d(c * 14, c * 2, 2, 2),
            norm(c * 2),
            nn.ReLU(inplace=True),
            BasicBlock(c * 2, c * 2, norm=norm),
        )

        self.Decoder2 = nn.Sequential(
            BasicBlock(c * 4, c * 2, norm=norm),
            nn.ConvTranspose3d(c * 2, c, 4, 2, padding=1),
            norm(c),
            nn.ReLU(inplace=True),
            BasicBlock(c, c, norm=norm),
        )

        self.Decoder3 = nn.Sequential(
            BasicBlock(c * 2, c, norm=norm),
            nn.Conv3d(c, 1, 3, 1, 1),
        )

    def forward(self, x: Tensor, rank=0) -> Tensor:
        if rank != 0:
            return self.forward2(x, rank - 1)
        # x = self.prepad(x)
        skip1 = self.input_layer1(x)  # 24, N
        skip2 = self.input_layer2(skip1)  # 48, N//2
        x = self.input_layer3(skip2)  # 192, N//4

        # 48, N//4; 96, N//8;
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)  # 48, N//4; 96, N//8
        x = [
            self.transition2[0](x[0]),  # 48, N//4
            self.transition2[1](x[1]),  # 96, N//8
            self.transition2[2](x[-1]),  # 192, N//16
        ]
        # 48, N//4; 96, N//8; 192, N//16
        x = self.stage3(x)

        _, _, t, h, w = x[0].shape  # b, 48, N//4
        x = torch.cat(
            [
                x[0],
                F.interpolate(x[1], size=(t, h, w)),
                F.interpolate(x[2], size=(t, h, w))
            ],
            1,
        )  # 336, N//4
        x = self.Decoder1(x)  # 48, N//2

        x = self.Decoder2(torch.cat([x, skip2], 1))  # 24, N
        x = self.Decoder3(torch.cat([x, skip1], 1))
        return torch.sigmoid(x)

    @torch.no_grad()
    def forward2(self, x: Tensor, rank=0, *args, **kwargs) -> Tensor:
        assert not self.training
        res = x

        if rank == 0:
            x = Conv3dChunked(self.input_layer1[0], 64)(x)
            x = self.input_layer1[1:3](x)
            x = self.input_layer1[3].chunk_conv_forward(x)
            skip1 = self.input_layer1[4].chunk_conv_forward(x)
            x = Conv3dChunked(self.input_layer2[0], 64)(skip1)
            x = self.input_layer2[1:3](x)
        elif rank == 1:
            if self.mode == 'iou':
                skip1 = self._input_layer12(x, 128, down=False)
            else:
                skip1 = self._input_layer1(x, 128, down=False)
            x = Conv3dChunked(self.input_layer2[0], 64)(skip1)
            x = self.input_layer2[1:3](x)
        else:
            if self.mode == 'iou':
                x = self._input_layer12(x, 128, down=True)
            else:
                x = self._input_layer1(x, 128, down=True)

        x = self.input_layer2[3].chunk_conv_forward(x)
        skip2 = self.input_layer2[4].chunk_conv_forward(x)

        # x = self.input_layer3(skip2)  # 192, N//4
        x = Conv3dChunked(self.input_layer3[0], 128)(skip2)
        x = self.input_layer3[1:3](x)
        x = self.input_layer3[3].chunk_conv_forward(x)
        x = self.input_layer3[4].chunk_conv_forward(x)

        # 48, N//4; 96, N//8;
        self.transition1[0][0] = Conv3dChunked(self.transition1[0][0], 32)
        self.transition1[1][0][0] = Conv3dChunked(self.transition1[1][0][0],
                                                  32)
        x = [trans(x) for trans in self.transition1]

        # Unchunked
        self.transition1[0][0] = self.transition1[0][0].ops
        self.transition1[1][0][0] = self.transition1[1][0][0].ops

        x = self.stage2(x)  # 48, N//4; 96, N//8

        x = [
            self.transition2[0](x[0]),  # 48, N//4
            self.transition2[1](x[1]),  # 96, N//8
            self.transition2[2](x[-1]),  # 192, N//16
        ]

        # 48, N//4; 96, N//8; 192, N//16
        x = self.stage3(x)

        _, _, t, h, w = x[0].shape  # b, 48, N//4
        x = torch.cat(
            [
                x[0],
                interpolate3d_chunked(x[1], 32, 2),
                interpolate3d_chunked(x[2], 32, 4),
            ],
            1,
        )  # 336, N//4
        x = ConvTranspose3dChunked(self.Decoder1[0], 64)(x)  # 48, N//2
        x = self.Decoder1[1:3](x)  # 48, N//2
        x = self.Decoder1[3].chunk_conv_forward(x)  # 48, N//2

        if rank == 0:
            x = self.Decoder2[0].chunk_conv_forward(torch.cat([x, skip2],
                                                              1))  # 24, N
            x = ConvTranspose3dChunked(self.Decoder2[1], 64)(x)
            x = self.Decoder2[2:4](x)
            x = self.Decoder2[4].chunk_conv_forward(x)
            x = self.Decoder3[0].chunk_conv_forward(torch.cat([x, skip1], 1))
            x = self.Decoder3[1](x)
        elif rank == 1:
            if self.mode == 'iou':
                x = self._decoder232(x, skip2, skip1=skip1)
            else:
                x = self._decoder23(x, skip2, skip1=skip1)
        else:
            if self.mode == 'iou':
                x = self._decoder232(x, skip2, inp=res)
            else:
                x = self._decoder23(x, skip2, inp=res)
        return torch.sigmoid(x)

    def load_state_dict(self, state_dict):
        new_state_dict = {}
        mode = 'iou'
        for k, v in state_dict.items():
            new_k = k.replace('.In.', '.')
            new_state_dict[new_k] = v
            if 'running_mean' in k or 'running_var' in k:
                mode = 'precision'
        if mode == 'precision' and self.mode == 'iou':
            print(
                f"The state_dict contains 'running_mean' indicating it is a `precision` model, but `iou` mode is required"
            )
        super(FaultSSL, self).load_state_dict(new_state_dict)

    def fuse_model(self):
        assert self.mode != 'iou', "Unsupport now!"
        if self.fused:
            return
        assert not self.training
        if not isinstance(self.input_layer1[1], nn.BatchNorm3d):
            self.fused = True
            return
        # fuse_modules(self.input_layer1, ['0', '1'], inplace=True)
        self.input_layer1[0] = ConvBatchNorm3d(self.input_layer1[0],
                                               self.input_layer1[1])
        self.input_layer1[1] = nn.Identity()
        self.input_layer1[3].fuse_model()
        self.input_layer1[4].fuse_model()

        # fuse_modules(self.input_layer2, ['0', '1'], inplace=True)
        self.input_layer2[0] = ConvBatchNorm3d(self.input_layer2[0],
                                               self.input_layer2[1])
        self.input_layer2[1] = nn.Identity()
        self.input_layer2[3].fuse_model()

        self.Decoder1[3].fuse_model()
        self.Decoder2[0].fuse_model()
        self.Decoder2[4].fuse_model()
        self.Decoder3[0].fuse_model()
        self.fused = True

    def unfuse_model(self):
        assert self.mode != 'iou', "Unsupport now!"
        if not self.fused:
            return
        assert not self.training
        conv1, norm1 = self.input_layer1[0].unfuse()
        self.input_layer1[0] = conv1
        self.input_layer1[1] = norm1
        self.input_layer1[3].unfuse_model()
        self.input_layer1[4].unfuse_model()

        conv2, norm2 = self.input_layer2[0].unfuse()
        self.input_layer2[0] = conv2
        self.input_layer2[1] = norm2
        self.input_layer2[3].unfuse_model()

        self.Decoder1[3].unfuse_model()
        self.Decoder2[0].unfuse_model()
        self.Decoder2[4].unfuse_model()
        self.Decoder3[0].unfuse_model()
        self.fused = False
