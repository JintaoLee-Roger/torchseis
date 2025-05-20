# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import itertools
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...ops.chunked.conv import Conv3dChunked, set_pad_as_zero


class BasicBlock(nn.Module):

    def __init__(self, inc: int, outc: int):
        super(BasicBlock, self).__init__()
        self.inc = inc
        self.outc = outc
        self.conv1 = nn.Conv3d(inc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(outc, outc, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x

    def chunk_forward(self, x: Tensor, padlist: list):
        x = self.conv1(x)
        x = set_pad_as_zero(x, padlist)
        x = self.relu1(x)
        x = self.conv2(x)
        x = set_pad_as_zero(x, padlist)
        x = self.relu2(x)
        return x

    def chuncked_conv_forward(self, x: Tensor, bsize: int = 128):
        x = Conv3dChunked(self.conv1, bsize)(x)
        x = self.relu1(x)
        x = Conv3dChunked(self.conv2, bsize)(x)
        x = self.relu2(x)
        return x


class FuseOps:

    def fuse_forward(
        self,
        x: Tensor,
        conv: BasicBlock,
        mode: str = 'down',
        onlydown: bool = False,
        y: Tensor = None,
        conv_last: nn.Conv3d = None,
        pool: nn.MaxPool3d = None,
        up: nn.Upsample = None,
    ) -> Tensor:
        b, c, d, h, w = x.shape
        pad = 2

        if mode == 'down':
            assert y is None and conv_last is None
            oc = conv.outc
            scale = 1
            if onlydown:
                oshape = (b, oc, d // 2, h // 2, w // 2)
            else:
                oshape = (b, oc, d, h, w)
        else:
            assert y is not None
            scale = 2
            oc = 1 if conv_last is not None else conv.outc
            oshape = (b, oc, d * 2, h * 2, w * 2)
            recompute = False
            if y.shape[1] == 1:
                recompute = True
                pad = 4
        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        bsize = 128
        nb_d = (d * scale + bsize - 1) // bsize
        nb_h = (h * scale + bsize - 1) // bsize
        nb_w = (w * scale + bsize - 1) // bsize

        for i, j, k in itertools.product(range(nb_d), range(nb_h),
                                         range(nb_w)):
            v_d0 = i * bsize
            v_h0 = j * bsize
            v_w0 = k * bsize

            v_d1 = min(v_d0 + bsize, d * scale)
            v_h1 = min(v_h0 + bsize, h * scale)
            v_w1 = min(v_w0 + bsize, w * scale)

            p_d0 = max(0, v_d0 - pad)
            p_h0 = max(0, v_h0 - pad)
            p_w0 = max(0, v_w0 - pad)

            p_d1 = min(d * scale, v_d1 + pad)
            p_h1 = min(h * scale, v_h1 + pad)
            p_w1 = min(w * scale, v_w1 + pad)

            padlist = [0, 0, 0, 0, 0, 0]

            if v_d1 == p_d1:
                padlist[5] = pad
            if v_h1 == p_h1:
                padlist[3] = pad
            if v_w1 == p_w1:
                padlist[1] = pad
            if v_d0 == p_d0:
                padlist[4] = pad
            if v_h0 == p_h0:
                padlist[2] = pad
            if v_w0 == p_w0:
                padlist[0] = pad

            r_d0 = (v_d0 - p_d0 + padlist[4])
            r_d1 = r_d0 + (v_d1 - v_d0)
            r_h0 = (v_h0 - p_h0 + padlist[2])
            r_h1 = r_h0 + (v_h1 - v_h0)
            r_w0 = (v_w0 - p_w0 + padlist[0])
            r_w1 = r_w0 + (v_w1 - v_w0)

            o_d0 = v_d0
            o_h0 = v_h0
            o_w0 = v_w0

            o_d1 = v_d1
            o_h1 = v_h1
            o_w1 = v_w1

            if mode == 'down' and onlydown:
                r_d0, r_d1 = r_d0 // 2, r_d1 // 2
                r_h0, r_h1 = r_h0 // 2, r_h1 // 2
                r_w0, r_w1 = r_w0 // 2, r_w1 // 2
                o_d0, o_d1 = o_d0 // 2, o_d1 // 2
                o_h0, o_h1 = o_h0 // 2, o_h1 // 2
                o_w0, o_w1 = o_w0 // 2, o_w1 // 2

            with torch.no_grad():
                if mode == 'down':
                    patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    patchi = F.pad(patchi, padlist, mode='constant', value=0)
                    patchi = conv.chunk_forward(patchi, padlist)
                    if onlydown:
                        patchi = pool(patchi)
                else:
                    patchi = x[:, :, p_d0 // 2:p_d1 // 2, p_h0 // 2:p_h1 // 2, p_w0 // 2:p_w1 // 2]
                    patchi = up(patchi)
                    patchi = F.pad(patchi, padlist, mode='constant', value=0)
                    patchj = y[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    patchj = F.pad(patchj, padlist, mode='constant', value=0)

                    if recompute:
                        patchj = self.conv1.chunk_forward(patchj, padlist)
                    patchi = torch.cat([patchi, patchj], dim=1)
                    patchi = conv.chunk_forward(patchi, padlist)
                    if conv_last is not None:
                        patchi = conv_last(patchi)

                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out
