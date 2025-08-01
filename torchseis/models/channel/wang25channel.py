"""
To cite this model:
```text
@article{wang2025cigchannel,
    author = {Wang, G. and Wu, X. and Zhang, W.},
    title = {cigChannel: a large-scale 3D seismic dataset with labeled paleochannels for advancing deep learning in seismic interpretation},
    journal = {Earth System Science Data},
    volume = {17},
    year = {2025},
    number = {7},
    pages = {3447--3471},
    url = {https://essd.copernicus.org/articles/17/3447/2025/},
    doi = {10.5194/essd-17-3447-2025}
}
```
"""
import torch
from torch import nn, Tensor
from ._wang25channel_block import BasicBlock, UpBlock, _FuseOps
from ...ops.chunked.conv import Conv3dChunked

class Wang25Channel(nn.Module, _FuseOps):

    def __init__(self, in_c=1, out_c=3, base=32):
        super(Wang25Channel, self).__init__()
        self.in_c = 1
        self.out_c = 3
        self.base = base

        self.dconv1 = BasicBlock(in_c, base)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.dconv2 = BasicBlock(base, base * 2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.dconv3 = BasicBlock(base * 2, base * 4)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.bridge = BasicBlock(base * 4, base * 8)

        self.up3 = UpBlock(base * 8)
        self.uconv3 = BasicBlock(base * 12, base * 4)

        self.up2 = UpBlock(base * 4)
        self.uconv2 = BasicBlock(base * 6, base * 2)

        self.up1 = UpBlock(base * 2)
        self.uconv1 = BasicBlock(base * 3, base)

        self.last_layer = nn.Sequential(
            nn.Conv3d(base, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor, rank: int = 0) -> Tensor:
        if rank > 0 and self.eval():
            return self.forward2(x, rank-1)
        d1 = self.dconv1(x)
        d2 = self.down1(d1)

        d2 = self.dconv2(d2)
        d3 = self.down2(d2)

        d3 = self.dconv3(d3)
        out = self.down3(d3)

        out = self.bridge(out)

        out = self.up3(out)
        out = torch.cat([out, d3], dim=1)
        out = self.uconv3(out)

        out = self.up2(out)
        out = torch.cat([out, d2], dim=1)
        out = self.uconv2(out)


        out = self.up1(out)
        out = torch.cat([out, d1], dim=1)
        out = self.uconv1(out)

        out = self.last_layer(out)

        return out

    def forward2(self, x: Tensor, rank=0) -> Tensor:
        """
        This method is a placeholder for compatibility with the original model's interface.
        It simply calls the forward method.
        """
        assert not self.training
        assert rank in [0, 1, 2]

        res = x
        if rank == 0:
            d1 = self.dconv1.chunked_conv_forward(x)
            d2 = self.down1(d1)
        elif rank == 1:
            d1 = self.down_fuse(x, self.dconv1, False)
            d2 = self.down1(d1)
        else:
            d2 = self.down_fuse(
                x,
                self.dconv1,
                onlydown=True,
                pool=self.down1,
            )

        if rank == 0:
            d2 = self.dconv2.chunked_conv_forward(d2)
        else:
            d2 = self.down_fuse(d2, self.dconv2, False)
        d3 = self.down1(d2)

        if rank == 0:
            d3 = self.dconv3.chunked_conv_forward(d3)
        else:
            d3 = self.down_fuse(d3, self.dconv3, False)
        out = self.down3(d3)

        if rank == 0:
            out = self.bridge.chunked_conv_forward(out)
        else:
            out = self.down_fuse(out, self.bridge, False)

        if rank == 0:
            out = self.up3.chunked_conv_forward(out)
            out = torch.cat([out, d3], dim=1)
            out = self.uconv3.chunked_conv_forward(out)
        else:
            out = self.up_fuse(
                out,
                d3,
                self.uconv3,
                self.up3,
            )

        if rank == 0:
            out = self.up2.chunked_conv_forward(out)
            out = torch.cat([out, d2], dim=1)
            out = self.uconv2.chunked_conv_forward(out)
        else:
            out = self.up_fuse(
                out,
                d2,
                self.uconv2,
                self.up2,
            )

        if rank == 0:
            out = self.up1.chunked_conv_forward(out)
            out = torch.cat([out, d1], dim=1)
            out = self.uconv1.chunked_conv_forward(out)
            out = Conv3dChunked(self.last_layer[0], 128)(out)
            out = self.last_layer[1:](out)
        elif rank == 1:
            out = self.up_fuse(
                out,
                d1,
                self.uconv1,
                self.up1,
            )
            out = Conv3dChunked(self.last_layer[0], 128)(out)
            out = self.last_layer[1:](out)
        else:
            out = self.up_fuse(
                out,
                res,
                self.uconv1,
                self.up1,
                self.last_layer,
            )

        return out
