# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU).
# All rights reserved.


import torch
from torch import nn, Tensor
from torch.nn import functional as F
import itertools
from ...ops.chunked.chunk_base import get_index
from ...ops.chunked.conv import set_pad_as_zero, Conv3dChunked


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module = None,
        norm: nn.Module = None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
    def _forward_conv_chunk(self, x):
        residual = x

        out = Conv3dChunked(self.conv1, 64)(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = Conv3dChunked(self.conv2, 64)(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = Conv3dChunked(self.conv3, 64)(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = Conv3dChunked(self.downsample[0], 64)(x)
            residual = self.downsample[1](residual)
            # residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, layers=[3, 4, 23, 3], norm=nn.BatchNorm3d):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layers = layers

        self.conv1 = nn.Conv3d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        down = nn.Sequential(
            nn.Conv3d(64, 64 * 4, 1, stride=1, bias=False),
            norm(64 * 4),
        )
        layer1 = [Bottleneck(64, 64, downsample=down, norm=norm)]
        for i in range(1, layers[0]):
            layer1.append(Bottleneck(64 * 4, 64, downsample=None, norm=norm))
        self.layer1 = nn.Sequential(*layer1)

        down = nn.Sequential(
            nn.Conv3d(64 * 4, 128 * 4, 1, stride=2, bias=False),
            norm(128 * 4),
        )
        layer2 = [
            Bottleneck(64 * 4, 128, stride=2, downsample=down, norm=norm)
        ]
        for i in range(1, layers[1]):
            layer2.append(Bottleneck(128 * 4, 128, downsample=None, norm=norm))
        self.layer2 = nn.Sequential(*layer2)

        down = nn.Sequential(
            nn.Conv3d(128 * 4, 256 * 4, 1, stride=2, bias=False),
            norm(256 * 4),
        )
        layer3 = [Bottleneck(128 * 4, 256, 2, 1, downsample=down, norm=norm)]
        for i in range(1, layers[2]):
            layer3.append(Bottleneck(256 * 4, 256, norm=norm))
        self.layer3 = nn.Sequential(*layer3)

        down = nn.Sequential(
            nn.Conv3d(256 * 4, 512 * 4, 1, stride=1, bias=False),
            norm(512 * 4),
        )
        layer4 = [Bottleneck(256 * 4, 512, 1, 2, downsample=down, norm=norm)]
        for i in range(1, layers[3]):
            layer4.append(Bottleneck(512 * 4, 512, 1, 4 * i, norm=norm))
        self.layer4 = nn.Sequential(*layer4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x) # 64, H/2
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x) # 64, H//4

        x = self.layer1(x) # 256, H//4
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat
    
    def _forward_chunk(self, x: Tensor, bsize=256) -> Tensor:
        b, c, d, h, w = x.shape
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        oshape = (b, 64, d // 4, h // 4, w // 4)
        pad = 8
        out = torch.zeros(oshape, dtype=x.dtype, device=x.device)
        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale=0.25)
            padlist2 = [f // 2 for f in padlist]
            with torch.no_grad():
                patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patchi = self.conv1(patchi) # the first conv layer need pad 4
                patchi = self.bn1(patchi)
                patchi = self.relu(patchi)
                patchi = set_pad_as_zero(patchi, padlist2)
                patchi = self.maxpool(patchi) # need pad 2, for the first layer: 4
                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]


        for i in range(self.layers[0]):
            out = self.layer1[i]._forward_conv_chunk(out)
        # out = self.layer1(out) # 256, H//4


        low_level_feat = out
        for i in range(self.layers[1]):
            out = self.layer2[i]._forward_conv_chunk(out)
        # out = self.layer2(out)

        for i in range(self.layers[2]):
            out = self.layer3[i]._forward_conv_chunk(out)

        for i in range(self.layers[3]):
            out = self.layer4[i]._forward_conv_chunk(out)
        # out = self.layer4(out)
        return out, low_level_feat


class ASPP(nn.Module):

    def __init__(self, norm=nn.BatchNorm3d):
        super(ASPP, self).__init__()
        inplanes = 2048
        planes = 256

        self.aspp1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 1, 1, 0, 1, bias=False),
            norm(planes),
            nn.ReLU(inplace=True),
        )

        self.aspp2 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 3, 1, 6, 6, bias=False),
            norm(planes),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 3, 1, 12, 12, bias=False),
            norm(planes),
            nn.ReLU(inplace=True),
        )

        self.aspp4 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 3, 1, 18, 18, bias=False),
            norm(planes),
            nn.ReLU(inplace=True),
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(inplanes, 256, 1, 1, bias=False),
            norm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv3d(1280, 256, 1, bias=False)
        self.bn1 = norm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5,
            size=x4.size()[2:],
            mode='trilinear',
            align_corners=True,
        )
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)
    
    def _forward_conv_chunk(self, x: Tensor) -> Tensor:
        x1 = Conv3dChunked(self.aspp1[0], 32)(x)
        x1 = self.aspp1[1:](x1)
        x2 = Conv3dChunked(self.aspp2[0], 32)(x)
        x2 = self.aspp2[1:](x2)
        x3 = Conv3dChunked(self.aspp3[0], 32)(x)
        x3 = self.aspp3[1:](x3)
        x4 = Conv3dChunked(self.aspp4[0], 32)(x)
        x4 = self.aspp4[1:](x4)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5,
            size=x4.size()[2:],
            mode='trilinear',
            align_corners=True,
        )
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = Conv3dChunked(self.conv1, 32)(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x) 


class Decoder(nn.Module):

    def __init__(self, num_classes=2, norm=nn.BatchNorm3d):
        super(Decoder, self).__init__()
        inplanes = 256
        self.conv1 = nn.Conv3d(inplanes, 48, 1, bias=False)
        self.norm1 = norm(48)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Sequential(
            nn.Conv3d(304, 256, 3, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv3d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv3d(256, num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x: Tensor, low_level_feat: Tensor) -> Tensor:
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.norm1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x,
            size=low_level_feat.size()[2:],
            mode='trilinear',
            align_corners=True,
        )

        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)

        return x

    def _forward_conv_chunk(self, x: Tensor, low_level_feat: Tensor) -> Tensor:
        low_level_feat = Conv3dChunked(self.conv1, 64)(low_level_feat)
        low_level_feat = self.norm1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(
            x,
            size=low_level_feat.size()[2:],
            mode='trilinear',
            align_corners=True,
        )

        x = torch.cat((x, low_level_feat), dim=1)

        x = Conv3dChunked(self.last_conv[0], 32)(x)
        x = self.last_conv[1:4](x)
        x = Conv3dChunked(self.last_conv[4], 32)(x)
        x = self.last_conv[5:](x)

        return x
