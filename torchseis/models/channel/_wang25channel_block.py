import itertools
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ...ops.chunked.conv import Conv3dChunked, set_pad_as_zero, ConvTranspose3dChunked
from ...ops.chunked.chunk_base import get_index

class BasicBlock(nn.Module):
    def __init__(self, inc: int, outc: int):
        super(BasicBlock, self).__init__()
        self.outc = outc
        self.conv1 = nn.Conv3d(inc, outc, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(outc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(outc, outc, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
    
    def chunk_forward(self, x: Tensor, padlist: list) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = set_pad_as_zero(x, padlist)
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = set_pad_as_zero(x, padlist)
        return x
    
    def chunked_conv_forward(self, x: Tensor, bsize: int = 64):
        x = Conv3dChunked(self.conv1, bsize)(x)
        x = self.relu1(self.bn1(x))
        x = Conv3dChunked(self.conv2, bsize)(x)
        x = self.relu2(self.bn2(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.tconv = nn.ConvTranspose3d(ch, ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm3d(ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def chunked_conv_forward(self, x: Tensor, bsize: int = 64):
        x = ConvTranspose3dChunked(self.tconv, bsize)(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def chunk_forward(self, x: Tensor, padlist: list) -> Tensor:
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = set_pad_as_zero(x, padlist)
        return x



class _FuseOps:

    def down_fuse(
        self,
        x: Tensor,
        conv: BasicBlock,
        onlydown: bool = False,
        pool: nn.MaxPool3d = None,
    ) -> Tensor:
        b, c, d, h, w = x.shape
        pad = 2

        oc = conv.outc
        if onlydown:
            oshape = (b, oc, d // 2, h // 2, w // 2)
            scale = 0.5
        else:
            oshape = (b, oc, d, h, w)
            scale = 1

        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        bsize = 128
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
            with torch.no_grad():
                patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patchi = conv.chunk_forward(patchi, padlist)
                if onlydown:
                    patchi = pool(patchi)

                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out
    
    def up_fuse(self, x: Tensor, y: Tensor, conv: BasicBlock, up: UpBlock, conv_last=None):
        """
        x = up(x)
        x = torch.cat([x, y], dim=1)
        x = conv(x)
        """
        b, c, d, h, w = x.shape
        pad = 3

        if conv_last is None:
            oc = conv.outc
        else:
            oc = self.out_c
        oshape = (b, oc, d*2, h*2, w*2)
        scale = 2

        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
            padlist2 = [f*2 for f in padlist]
            with torch.no_grad():
                patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patchi = up.chunk_forward(patchi, padlist2)

                patchj = y[:, :, p_d0*2:p_d1*2, p_h0*2:p_h1*2, p_w0*2:p_w1*2]
                patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                if conv_last is not None:
                    patchj = self.dconv1.chunk_forward(patchj, padlist2)

                patchi = torch.cat([patchi, patchj], dim=1)
                patchi = conv.chunk_forward(patchi, padlist2)

                if conv_last is not None:
                    patchi = conv_last(patchi)

                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out

