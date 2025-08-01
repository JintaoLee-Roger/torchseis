"""
Bi21RGT3d model is a convolutional U-Net with attention mechanism (I'm not sure whether that could be called 'attention').

To cite this model:
```text
@article{bi2021deep,
    title={Deep relative geologic time: A deep learning method for simultaneously interpreting 3-D seismic horizons and faults},
    author={Bi, Zhengfa and Wu, Xinming and Geng, Zhicheng and Li, Haishan},
    journal={Journal of Geophysical Research: Solid Earth},
    volume={126},
    number={9},
    pages={e2021JB021882},
    year={2021},
    publisher={Wiley Online Library}
}
```
"""

from torch import nn, Tensor
from torch.nn import functional as F
from ._bi21rgt3d_blocks import BasicBlock, UP, _FuseOps
import time

class Bi21RGT3d(nn.Module, _FuseOps):
    """
    Implements Bi21RGT3d model from
    `"Deep relative geologic time: A deep learning method for simultaneously interpreting 3-D seismic horizons and faults"
    <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB021882>`_.

    Cost tested on a H20, (precision/precision-fuse/iou)
    ---------------------
    | rank | shape              | memory   | time   | GPU        |
    |------|--------------------|----------|--------|------------|
    | 0    | (256, 256, 256)    | 13.44 GB | 17.2s  | H20 96 GB  |
    | 0    | (512, 512, 512)    | 85.36 GB | 360.4s | H20 96 GB  |
    |------|--------------------|----------|--------|------------|
    | 1    | (512, 512, 512)    | 26.41 GB | 272.1s | V100 32 GB  |
    | 1    | (768, 768, 768)    | 86.38 GB | 1256 s | H20 96 GB |

    """
    def __init__(self, fused=False, norm=nn.InstanceNorm3d):
        super(Bi21RGT3d, self).__init__()

        self.fused = fused
        # encoder layers
        self.layer0 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=7, stride=1, padding=3, bias=False),
            norm(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            nn.Conv3d(16, 16, 3, stride=1, padding=1, bias=False),
            norm(16),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
        )

        downsample = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=2, bias=False),
            norm(64),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, stride=2, downsample=downsample, dilation=(1, 1), norm=norm),
            BasicBlock(64, 64, stride=1, downsample=None, dilation=(1, 1), norm=norm),
        )

        downsample = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=False),
            norm(128),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=downsample, dilation=(1, 1), norm=norm),
            BasicBlock(128, 128, stride=1, downsample=None, dilation=(1, 1), norm=norm),
        )
        downsample = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=1, bias=False),
            norm(256),
        )
        self.layer5 = nn.Sequential(
            BasicBlock(128, 256, stride=1, downsample=downsample, dilation=(2, 2), norm=norm),
            BasicBlock(256, 256, stride=1, downsample=None, dilation=(2, 2), norm=norm),
        )

        self.layer6 = None 

        self.layer7 = nn.Sequential(
            nn.Conv3d(256, 512, 3, stride=1, padding=2, bias=False, dilation=2),
            norm(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, stride=1, padding=2, bias=False, dilation=2),
            norm(512),
            nn.ReLU(inplace=True),
        )

        self.layer8 = nn.Sequential(
            nn.Conv3d(512, 512, 3, stride=1, padding=1, bias=False, dilation=1),
            norm(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, 3, stride=1, padding=1, bias=False, dilation=1),
            norm(512),
            nn.ReLU(inplace=True),
        )

        # decoder layers
        self.conv = nn.Conv3d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn = nn.InstanceNorm3d(512)

        self.up3 = UP(512, 64, 256)
        self.up4 = UP(256, 32, 128)
        self.up5 = UP(128, 16, 16)

        self.out_layer = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
        )

    def encoder(self, x: Tensor):
        x_block0 = self.layer0(x) # 1->16, H->H

        x = self.layer1(x_block0) # 16->16, H->H
        x_block1 = self.layer2(x) # 16->32, H->H/2
        x_block2 = self.layer3(x_block1) # 32->64, H/2->H/4
        x_block3 = self.layer4(x_block2) # 64->128, H/4->H/8

        x = self.layer5(x_block3) # 128->256, H/8->H/8
        x = self.layer7(x) # 256->512, H/8->H/8
        x_block4 = self.layer8(x) # 512->512, H/8->H/8

        return x_block0, x_block1, x_block2, x_block4

    def decoder(self, x_block0, x_block1, x_block2, x_block4):
        out = F.relu(self.bn(self.conv(x_block4))) # 512->512, H/8->H/8
        out = self.up3(out, x_block2, [x_block2.size(2), x_block2.size(3), x_block2.size(4)]) # 512,64->256, H/8,H/4->H/4
        out = self.up4(out, x_block1, [x_block1.size(2), x_block1.size(3), x_block1.size(4)]) # 256,32->128, H/4,H/2->H/2
        out = self.up5(out, x_block0, [x_block0.size(2), x_block0.size(3), x_block0.size(4)]) # 128,16->16,  H/2,H  ->H
        return self.out_layer(out) # 16->1, H->H

    def forward(self, x: Tensor) -> Tensor:
        x_block0 = self.layer0(x) # 1->16, H->H

        x = self.layer1(x_block0) # 16->16, H->H
        x_block1 = self.layer2(x) # 16->32, H->H/2
        x_block2 = self.layer3(x_block1) # 32->64, H/2->H/4
        x_block3 = self.layer4(x_block2) # 64->128, H/4->H/8

        x = self.layer5(x_block3) # 128->256, H/8->H/8
        x = self.layer7(x) # 256->512, H/8->H/8
        x_block4 = self.layer8(x) # 512->512, H/8->H/8

        out = F.relu(self.bn(self.conv(x_block4))) # 512->512, H/8->H/8
        out = self.up3(out, x_block2, [x_block2.size(2), x_block2.size(3), x_block2.size(4)]) # 512,64->256, H/8,H/4->H/4
        out = self.up4(out, x_block1, [x_block1.size(2), x_block1.size(3), x_block1.size(4)]) # 256,32->128, H/4,H/2->H/2
        out = self.up5(out, x_block0, [x_block0.size(2), x_block0.size(3), x_block0.size(4)]) # 128,16->16,  H/2,H  ->H
        out = self.out_layer(out) # 16->1, H->H

        return out

    
    def forward2(self, x: Tensor, rank=None) -> Tensor:
        res = x
        x_block1 = self._layer012(x, 64)
        x_block2 = self.layer3(x_block1) # 32->64, H/2->H/4
        x_block3 = self.layer4(x_block2) # 64->128, H/4->H/8

        x = self.layer5(x_block3) # 128->256, H/8->H/8
        x = self.layer7(x) # 256->512, H/8->H/8
        x_block4 = self.layer8(x) # 512->512, H/8->H/8

        out = F.relu(self.bn(self.conv(x_block4))) # 512->512, H/8->H/8
        out = self.up3(out, x_block2, [x_block2.size(2), x_block2.size(3), x_block2.size(4)]) # 512,64->256, H/8,H/4->H/4
        out = self.up4(out, x_block1, [x_block1.size(2), x_block1.size(3), x_block1.size(4)]) # 256,32->128, H/4,H/2->H/2
        out = self._up5(out, res, 64)
        return self.out_layer[1](out)