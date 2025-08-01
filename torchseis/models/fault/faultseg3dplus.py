"""
FaultSeg3DPlus model is a 3D convolutional U-Net for seismic fault segmentation.

To cite this model:
```text
@article{li2024faultseg3d,
    title={FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation},
    author={Li, You and Wu, Xinming and Zhu, Zhenyu and Ding, Jicai and Wang, Qingzhen},
    journal={Geophysics},
    volume={89},
    number={5},
    pages={N77--N91},
    year={2024},
    publisher={Society of Exploration Geophysicists},
    doi={10.1190/geo2022-0778.1},
    url={https://library.seg.org/doi/10.1190/geo2022-0778.1}
}
```
"""

import torch
from torch import nn, Tensor
from ._faultseg_modules import BasicBlock, FuseOps
from ...ops.chunked.conv import Conv3dChunked
from ...ops.chunked.interpolate import UpsampleChunked

__all__ = ['FaultSeg3dPlus']


class FaultSeg3dPlus(nn.Module, FuseOps):
    """
    Implements FaultSeg3D model from 
    `"FaultSeg3D plus: A comprehensive study on evaluating and improving CNN-based seismic fault segmentation"
    <https://library.seg.org/doi/10.1190/geo2022-0778.1>`_.

    | rank | shape              | memory   | time   | GPU        |
    |------|--------------------|----------|--------|------------|
    | 0    | (384, 384, 256)    | 23.58 GB | 11.70s | RTX 24 GB  |
    | 0    | (384, 384, 288)    | 26.42 GB | 12.76s | V100 32 GB |
    | 0    | (416, 384, 384)    | 37.78 GB | 18.94s | A100 40 GB |
    | 0    | (576, 512, 512)    | 75.96 GB | 97.26s | A100 80 GB |
    | 0    | (672, 640, 512)    | 93.97 GB | 143.6s | H20 96 GB  |
    |------|--------------------|----------|--------|------------|
    | 1    | (384, 384, 288)    | 23.27 GB | 0.73s  | RTX 24 GB  |
    | 1    | (384, 384, 384)    | 30.52 GB | 0.90s  | V100 32 GB |
    | 1    | (512, 384, 384)    | 39.94 GB | 1.17s  | A100 40 GB |
    | 1    | (576, 512, 512)    | 78.18 GB | 2.20s  | A100 80 GB |
    | 1    | (672, 640, 512)    | 93.97 GB | 3.40s  | H20 96 GB  |
    |------|--------------------|----------|--------|------------|
    | 2    | (704, 512, 512)    | 23.61 GB | 2.39s  | RTX 24 GB  |
    | 2    | (768, 672, 512)    | 31.23 GB | 3.31s  | V100 32 GB |
    | 2    | (768, 768, 576)    | 38.96 GB | 4.22s  | A100 40 GB |
    | 2    | (1024, 960, 768)   | 79.86 GB | 8.98s  | A100 80 GB |
    | 2    | (1024, 1024, 864)  | 94.50 GB | 10.70s | H20 96 GB  |
    |------|--------------------|----------|--------|------------|
    | 3    | (738, 704, 704)    | 23.57 GB | 5.18s  | RTX 24 GB  |
    | 3    | (896, 800, 768)    | 31.84 GB | 7.66s  | V100 32 GB |
    | 3    | (1024, 896, 768)   | 39.83 GB | 9.62s  | A100 40 GB |
    | 3    | (1024, 1024, 1024) | 56.75 GB | 14.56s | - |
    | 3    | (1536, 1088, 1024) | 79.66 GB | 23.11s | A100 80 GB |
    | 3    | (1536, 1280, 1152) | 94.77 GB | 30.51s | H20 96 GB  |

    """

    def __init__(self, base: int = 32):
        super(FaultSeg3dPlus, self).__init__()
        self.need_padding = 2**4
        self.name = 'FaultSeg3dPlus'

        nf1 = base
        nf2 = nf1 * 2  # 32
        nf3 = nf2 * 2  # 64
        nf4 = nf3 * 2  # 128
        nf5 = nf4 * 2  # 256

        self.conv1 = BasicBlock(1, nf1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = BasicBlock(nf1, nf2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = BasicBlock(nf2, nf3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = BasicBlock(nf3, nf4)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = BasicBlock(nf4, nf5)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = BasicBlock(nf5 + nf4, nf4)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = BasicBlock(nf4 + nf3, nf3)

        self.up8 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = BasicBlock(nf3 + nf2, nf2)

        self.up9 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9 = BasicBlock(nf2 + nf1, nf1)

        self.conv10 = nn.Conv3d(nf1, 1, kernel_size=1)

    def forward(self, x: Tensor, rank=0) -> Tensor:
        if rank > 0:
            return self.forward2(x, rank - 1)
        encoder_features = []

        enc1 = self.conv1(x)
        encoder_features.append(enc1)
        enc2 = self.pool1(enc1)

        enc2 = self.conv2(enc2)
        encoder_features.append(enc2)
        enc3 = self.pool2(enc2)

        enc3 = self.conv3(enc3)
        encoder_features.append(enc3)
        enc4 = self.pool3(enc3)

        enc4 = self.conv4(enc4)
        encoder_features.append(enc4)
        out = self.pool4(enc4)

        out = self.conv5(out)

        out = self.up6(out)
        out = torch.cat([out, encoder_features[3]], dim=1)
        out = self.conv6(out)

        out = self.up7(out)
        out = torch.cat([out, encoder_features[2]], dim=1)
        out = self.conv7(out)

        out = self.up8(out)
        out = torch.cat([out, encoder_features[1]], dim=1)
        out = self.conv8(out)

        out = self.up9(out)
        out = torch.cat([out, encoder_features[0]], dim=1)
        out = self.conv9(out)

        out = self.conv10(out)

        return torch.sigmoid(out)

    def forward2(self, x: Tensor, rank: int = 0) -> Tensor:
        """
        rank 1: (1024, 1024, 768), 94.6GB
        rank 2: (2048, 1024, 1024) 95 GB
        """
        assert not self.training
        assert rank in [0, 1, 2]
        res = x
        encoder_features = []

        if rank == 0:
            enc1 = self.conv1.chunked_conv_forward(x)
            encoder_features.append(enc1)
            enc2 = self.pool1(enc1)
        elif rank == 1:
            enc1 = self.fuse_forward(x, self.conv1, mode='down')
            encoder_features.append(enc1)
            enc2 = self.pool1(enc1)
        elif rank == 2:
            enc2 = self.fuse_forward(
                x,
                self.conv1,
                mode='down',
                onlydown=True,
                pool=self.pool1,
            )
            encoder_features.append(None)

        if rank == 0:
            enc2 = self.conv2.chunked_conv_forward(enc2)
        else:
            enc2 = self.fuse_forward(enc2, self.conv2, mode='down')
        encoder_features.append(enc2)
        enc3 = self.pool2(enc2)

        if rank == 0:
            enc3 = self.conv3.chunked_conv_forward(enc3)
        else:
            enc3 = self.fuse_forward(enc3, self.conv3, mode='down')
        encoder_features.append(enc3)
        out = self.pool3(enc3)

        if rank == 0:
            out = self.conv4.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(out, self.conv4, mode='down')
        encoder_features.append(out)
        out = self.pool4(out)

        if rank == 0:
            out = self.conv5.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(out, self.conv5, mode='down')

        if rank == 0:
            out = UpsampleChunked(self.up6, 64)(out)
            out = torch.cat([out, encoder_features[3]], dim=1)
            out = self.conv6.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(
                out,
                self.conv6,
                mode='up',
                y=encoder_features[3],
                up=self.up6,
            )

        if rank == 0:
            out = UpsampleChunked(self.up7, 64)(out)
            out = torch.cat([out, encoder_features[2]], dim=1)
            out = self.conv7.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(
                out,
                self.conv7,
                mode='up',
                y=encoder_features[2],
                up=self.up7,
            )

        if rank == 0:
            out = UpsampleChunked(self.up8, 64)(out)
            out = torch.cat([out, encoder_features[1]], dim=1)
            out = self.conv8.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(
                out,
                self.conv8,
                mode='up',
                y=encoder_features[1],
                up=self.up8,
            )

        if rank == 0:
            out = UpsampleChunked(self.up9, 64)(out)
            out = torch.cat([out, encoder_features[0]], dim=1)
            out = self.conv9.chunked_conv_forward(out)
            out = Conv3dChunked(self.conv10, 64)(out)
        elif rank == 1:
            out = self.fuse_forward(
                out,
                self.conv9,
                mode='up',
                y=encoder_features[0],
                conv_last=self.conv10,
                up=self.up9,
            )
        elif rank == 2:
            out = self.fuse_forward(
                out,
                self.conv9,
                mode='up',
                y=res,
                conv_last=self.conv10,
                up=self.up9,
            )

        return torch.sigmoid(out)


if __name__ == "__main__":

    model = FaultSeg3dPlus()
