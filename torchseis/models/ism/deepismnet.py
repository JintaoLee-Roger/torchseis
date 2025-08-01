"""

To cite this model:
```text
@article{bi2022deepismnet,
    title={DeepISMNet: Three-dimensional implicit structural modeling with convolutional neural network},
    author={Bi, Zhengfa and Wu, Xinming and Li, Zhaoliang and Chang, Dekuan and Yong, Xueshan},
    journal={Geoscientific Model Development Discussions},
    volume={2022},
    pages={1--28},
    year={2022},
    publisher={G{\"o}ttingen, Germany}
}
```
"""

import torch
from torch import nn, Tensor
from ._deepismnet_block import _FuseOps, BasicBlock, UpConv, UP
from ...ops.chunked.conv import Conv3dChunked


class DeepISMNet(nn.Module, _FuseOps):

    def __init__(self, use_norm: bool = False):
        super(DeepISMNet, self).__init__()

        in_ch = 2 if not use_norm else 5
        self.in_c = in_ch 
        self.out_c = 1

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv1 = BasicBlock(3, in_ch, 63, 51)
        self.conv2 = BasicBlock(3, 51, 127, 102)
        self.conv3 = BasicBlock(3, 102, 255, 204, True)
        self.conv4 = BasicBlock(3, 204, 511, 409, True)
        self.conv5 = BasicBlock(3, 409, 1023, 819, True)

        self.up5 = UP(ch_in=819, ch_out=409)
        self.up_conv5 = UpConv(ch_in=409 + 409, ch_out=409)

        self.up4 = UP(ch_in=409, ch_out=204)
        self.up_conv4 = UpConv(ch_in=204 + 204, ch_out=204)

        self.up3 = UP(ch_in=204, ch_out=102)
        self.up_conv3 = UpConv(ch_in=102 + 102, ch_out=102)

        self.up2 = UP(ch_in=102, ch_out=51)
        self.up_conv2 = UpConv(ch_in=51 + 51, ch_out=51)

        self.conv_1x1 = nn.Conv3d(51, 1, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        out = self.maxpool(x4)
        out = self.conv5(out)

        out = self.up5(out, tuple(x4.shape[2:]))
        out = torch.cat((x4, out), dim=1)
        out = self.up_conv5(out)

        out = self.up4(out, tuple(x3.shape[2:]))
        out = torch.cat((x3, out), dim=1)
        out = self.up_conv4(out)

        out = self.up3(out, tuple(x2.shape[2:]))
        out = torch.cat((x2, out), dim=1)
        out = self.up_conv3(out)

        out = self.up2(out, tuple(x1.shape[2:]))
        out = torch.cat((x1, out), dim=1)
        out = self.up_conv2(out)

        out = self.conv_1x1(out)

        return out

    def forward2(self, x: Tensor, rank: int = 0) -> Tensor:
        assert not self.training
        assert rank in [0, 1, 2]

        res = x
        if rank == 0:
            x1 = self.conv1.chunked_conv_forward(x)
            x2 = self.maxpool(x1)
        elif rank == 1:
            x1 = self.down_fuse(x, self.conv1, False)
            x2 = self.maxpool(x1)
        else:
            x2 = self.down_fuse(
                x,
                self.conv1,
                onlydown=True,
                pool=self.maxpool,
            )

        if rank == 0:
            x2 = self.conv2.chunked_conv_forward(x2)
        else:
            x2 = self.down_fuse(x2, self.conv2, False)
        x3 = self.maxpool(x2)

        if rank == 0:
            x3 = self.conv3.chunked_conv_forward(x3)
        else:
            x3 = self.down_fuse(x3, self.conv3, False)

        x4 = self.maxpool(x3)

        if rank == 0:
            x4 = self.conv4.chunked_conv_forward(x4)
        else:
            x4 = self.down_fuse(x4, self.conv4, False)
        out = self.maxpool(x4)

        if rank == 0:
            out = self.conv5.chunked_conv_forward(out)
        else:
            out = self.down_fuse(out, self.conv5, False)

        if rank == 0:
            out = self.up5.chunked_conv_forward(out, x4.shape[2:])
            out = torch.cat([x4, out], dim=1)
            out = self.up_conv5.chunked_conv_forward(out)
        else:
            out = self.up_fuse(
                out,
                x4,
                self.up_conv5,
                self.up5,
            )

        if rank == 0:
            out = self.up4.chunked_conv_forward(out, x3.shape[2:])
            out = torch.cat([x3, out], dim=1)
            out = self.up_conv4.chunked_conv_forward(out)
        else:
            out = self.up_fuse(
                out,
                x3,
                self.up_conv4,
                self.up4,
            )

        if rank == 0:
            out = self.up3.chunked_conv_forward(out, x2.shape[2:])
            out = torch.cat([x2, out], dim=1)
            out = self.up_conv3.chunked_conv_forward(out)
        else:
            out = self.up_fuse(
                out,
                x2,
                self.up_conv3,
                self.up3,
            )

        if rank == 0:
            out = self.up2.chunked_conv_forward(out, x1.shape[2:])
            out = torch.cat([x1, out], dim=1)
            out = self.up_conv2.chunked_conv_forward(out)
            out = Conv3dChunked(self.conv_1x1, 128)(out)
        elif rank == 1:
            out = self.up_fuse(
                out,
                x1,
                self.up_conv2,
                self.up2,
            )
            out = Conv3dChunked(self.conv_1x1, 128)(out)
        else:
            out = self.up_fuse(
                out,
                res,
                self.up_conv2,
                self.up2,
                self.conv_1x1,
            )

        return out
