"""
FaultSeg3D model is a 3D convolutional U-Net for seismic fault segmentation.

To cite this model:
```text
@article{wu2019faultseg3d,
    title={FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation},
    author={Wu, Xinming and Liang, Luming and Shi, Yunzhi and Fomel, Sergey},
    journal={Geophysics},
    volume={84},
    number={3},
    pages={IM35--IM45},
    year={2019},
    publisher={Society of Exploration Geophysicists},
    doi={10.1190/geo2018-0646.1},
    url={https://library.seg.org/doi/10.1190/geo2018-0646.1}
}
```
"""

import torch
from torch import nn, Tensor
from ._faultseg_modules import BasicBlock, FuseOps
from ...ops.chunked.conv import Conv3dChunked
from ...ops.chunked.interpolate import UpsampleChunked

__all__ = ['FaultSeg3d']


class FaultSeg3d(nn.Module, FuseOps):
    """
    Implements FaultSeg3D model from
    `"FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation"
    <https://library.seg.org/doi/10.1190/geo2018-0646.1>`_.

    The original implementation (in Keras) is implemented by Xinming Wu, see: `https://github.com/xinwucwp/faultSeg/blob/master/unet3.py`.

    The following table shows the memory and time cost for different input shapes (tested on H20 96GB GPU, half precision).

    | Rank | Input Shape        | Memory   | Time    | GPU              |
    |------|-------------------|----------|---------|-------------------|
    | 0    | (512, 384, 384)   | 23.79 GB | 6.01s   | RTX    24 GB      |
    | 0    | (512, 512, 384)   | 31.51 GB | 8.05s   | V100   32 GB      |
    | 0    | (512, 512, 512)   | 38.80 GB | 17.01s  | A100   40 GB      |
    | 0    | (768, 704, 576)   | 79.17 GB | 49.99s  | A100   80 GB      |
    | 0    | (768, 768, 736)   | 93.04 GB | 70.35s  | H20    96 GB      |
    |------|-------------------|----------|---------|-------------------|
    | 1    | (512, 384, 384)   | 21.55 GB | 1.06s   | RTX    24 GB      |
    | 1    | (512, 512, 416)   | 30.73 GB | 1.42s   | V100   32 GB      |
    | 1    | (544, 512, 512)   | 39.71 GB | 1.76s   | A100   40 GB      |
    | 1    | (768, 768, 512)   | 76.80 GB | 3.52s   | A100   80 GB      |
    | 1    | (768, 768, 736)   | 93.04 GB | 5.19s   | H20    96 GB      |
    |------|-------------------|----------|---------|-------------------|
    | 2    | (768, 768, 704)   | 23.84 GB | 2.54s   | RTX    24 GB      |
    | 2    | (960, 768, 768)   | 31.45 GB | 3.38s   | V100   32 GB      |
    | 2    | (1024, 896, 768)  | 38.92 GB | 4.13s   | A100   40 GB      |
    | 2    | (1024, 1024, 1024)| 58.86 GB | 6.20s   |                   |
    | 2    | (1280, 1088, 1024)| 77.89 GB | 8.16s   | A100   80 GB      |
    | 2    | (1280, 1280, 1088)| 93.83 GB | 10.14s  | H20    96 GB      |
    |------|-------------------|----------|---------|-------------------|
    | 3    | (1024, 1024, 832) | 23.08 GB | 6.06s   | RTX    24 GB      |
    | 3    | (1024, 1024, 1024)| 26.86 GB | 7.26s   |                   |
    | 3    | (1152, 1088, 1024)| 31.94 GB | 8.77s   | V100   32 GB      |
    | 3    | (1216, 1152, 1152)| 39.94 GB | 10.95s  | A100   40 GB      |
    | 3    | (2048, 1536, 1024)| 78.86 GB | 21.42s  | A100   80 GB      |
    | 3    | (1600, 1536, 1536)| 92.27 GB | 25.09s  | H20    96 GB      |
    """

    def __init__(self, in_channels: int = 1, version: str = '2020'):
        super(FaultSeg3d, self).__init__()

        self.need_padding = 2**3
        self.name = 'FaultSeg3d'

        if version == '2018':
            ch_bottleneck = 512
        elif version == '2020':
            ch_bottleneck = 128
        else:
            raise ValueError(f'Invalid version: {version}')

        self.conv1 = BasicBlock(in_channels, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = BasicBlock(16, 32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = BasicBlock(32, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = BasicBlock(64, ch_bottleneck)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = BasicBlock(ch_bottleneck + 64, 64)

        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = BasicBlock(64 + 32, 32)

        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = BasicBlock(32 + 16, 16)

        self.conv8 = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x: Tensor, rank: int = 0) -> Tensor:
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
        out = self.pool3(enc3)

        out = self.conv4(out)

        out = self.up5(out)
        # NOTE: out is in the first position
        out = torch.cat([out, encoder_features[2]], dim=1)
        out = self.conv5(out)

        out = self.up6(out)
        out = torch.cat([out, encoder_features[1]], dim=1)
        out = self.conv6(out)

        out = self.up7(out)
        out = torch.cat([out, encoder_features[0]], dim=1)
        out = self.conv7(out)
        out = torch.sigmoid(self.conv8(out))

        return out

    def forward2(self, x: Tensor, rank: int = 0) -> Tensor:
        """
        rank 2: (1536, 1536, 1536) 91 GB
        """
        assert not self.training
        res = x
        encoder_features = []
        assert rank in [0, 1, 2]

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

        if rank == 0:
            out = UpsampleChunked(self.up5, 64)(out)
            out = torch.cat([out, encoder_features[2]], dim=1)
            out = self.conv5.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(
                out,
                self.conv5,
                mode='up',
                y=encoder_features[2],
                up=self.up5,
            )

        if rank == 0:
            out = UpsampleChunked(self.up6, 128)(out)
            out = torch.cat([out, encoder_features[1]], dim=1)
            out = self.conv6.chunked_conv_forward(out)
        else:
            out = self.fuse_forward(
                out,
                self.conv6,
                mode='up',
                y=encoder_features[1],
                up=self.up6,
            )

        if rank == 0:
            out = UpsampleChunked(self.up7, 128)(out)
            out = torch.cat([out, encoder_features[0]], dim=1)
            out = self.conv7.chunked_conv_forward(out)
            out = Conv3dChunked(self.conv8, 128)(out)
        elif rank == 1:
            out = self.fuse_forward(
                out,
                self.conv7,
                mode='up',
                y=encoder_features[0],
                conv_last=self.conv8,
                up=self.up7,
            )
        else:
            out = self.fuse_forward(
                out,
                self.conv7,
                mode='up',
                y=res,
                conv_last=self.conv8,
                up=self.up7,
            )
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = FaultSeg3d()
