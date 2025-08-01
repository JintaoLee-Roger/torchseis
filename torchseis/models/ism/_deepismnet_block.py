import itertools
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from ...ops.chunked.conv import Conv3dChunked, set_pad_as_zero
from ...ops.chunked.chunk_base import get_index
from ...ops.chunked.interpolate import trilinear_interpolate_align_corners


class hswish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):

    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_size,
                      in_size // reduction,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm3d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_size // reduction,
                      in_size,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm3d(in_size),
            hsigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

    def chunked_conv_forward(self, x: Tensor, bsize: int = 128):
        y = self.se[0](x)
        y = Conv3dChunked(self.se[1], bsize)(y)
        y = self.se[2:4](y)
        y = Conv3dChunked(self.se[4], bsize)(y)
        y = self.se[5:](y)
        return x * y
    
    def chunk_forward(self, x: Tensor, padlist: list, indices) -> Tensor:
        # y = self.se[0](x)
        # y = self.se[1](y)
        # y = self.se[2:4](y)
        # y = self.se[4](y)
        # y = self.se[5:](y)
        x = x * self.se(x[indices])
        x = set_pad_as_zero(x, padlist)
        return x


class BasicBlock(nn.Module):
    # pad = 1

    def __init__(self,
                 kernel_size,
                 in_size,
                 expand_size,
                 out_size,
                 semodule=False):
        super(BasicBlock, self).__init__()
        stride = 1
        self.outc = out_size
        if semodule:
            self.se = SeModule(out_size)
        else:
            self.se = None

        self.conv1 = nn.Conv3d(in_size, expand_size, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm3d(expand_size)
        self.nolinear1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(expand_size,
                               expand_size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size // 2,
                               groups=expand_size,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(expand_size)
        self.nolinear2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(expand_size, out_size, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_size, out_size, 1, 1, 0, bias=False),
                nn.BatchNorm3d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x)
        return out

    def chunked_conv_forward(self, x: Tensor, bsize: int = 128):
        res = x
        x = Conv3dChunked(self.conv1, bsize)(x)
        x = self.nolinear1(self.bn1(x))
        x = Conv3dChunked(self.conv2, bsize)(x)
        x = self.nolinear2(self.bn2(x))
        x = Conv3dChunked(self.conv3, bsize)(x)
        x = self.bn3(x)
        if self.se is not None:
            x = self.se.chunked_conv_forward(x, bsize)

        if len(self.shortcut) > 0:
            y = Conv3dChunked(self.shortcut[0], bsize)(res)
            x = x + self.shortcut[1](y)

        return x
    
    def chunk_forward(self, x: Tensor, padlist: list, indices) -> Tensor:
        res = x
        x = self.nolinear1(self.bn1(self.conv1(x)))
        x = set_pad_as_zero(x, padlist)
        x = self.nolinear2(self.bn2(self.conv2(x)))
        x = set_pad_as_zero(x, padlist)
        x = self.bn3(self.conv3(x))
        x = set_pad_as_zero(x, padlist)

        if self.se is not None:
            x = self.se.chunk_forward(x, padlist, indices)

        if len(self.shortcut) > 0:
            y = self.shortcut(res)
            y = set_pad_as_zero(y, padlist)
            x = x + y
        return x


class UpConv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.outc = ch_out
        self.layer1 = nn.Sequential(
            nn.Conv3d(ch_in, ch_in, 3, 1, 1, groups=ch_in, bias=False),
            nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in, ch_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, 3, 1, 1, groups=ch_out, bias=False),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def chunked_conv_forward(self, x: Tensor, bsize: int = 128):
        x = Conv3dChunked(self.layer1[0], bsize)(x)
        x = self.layer1[1:3](x)
        x = Conv3dChunked(self.layer1[3], bsize)(x)
        x = self.layer1[4:](x)
        x = Conv3dChunked(self.layer2[0], bsize)(x)
        x = self.layer2[1:3](x)
        x = Conv3dChunked(self.layer2[3], bsize)(x)
        x = self.layer2[4:](x)
        return x
    
    def chunk_forward(self, x: Tensor, padlist: list) -> Tensor:
        x = self.layer1[:3](x)
        x = set_pad_as_zero(x, padlist)
        x = self.layer1[3:](x)
        x = set_pad_as_zero(x, padlist)
        x = self.layer2[:3](x)
        x = set_pad_as_zero(x, padlist)
        x = self.layer2[3:](x)
        x = set_pad_as_zero(x, padlist)
        return x


class UP(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(UP, self).__init__()
        self.outc = ch_out
        self.up = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 1, 1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        x = self.up(x)
        return x

    def chunked_conv_forward(self, x: Tensor, size: tuple, bsize: int = 128):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        x = Conv3dChunked(self.up[0], bsize)(x)
        x = self.up[1:](x)
        return x
    
    def chunk_forward(self, x: Tensor, padlist: list) -> Tensor:
        # NOTE: We don't interpolate here, because the full size is not known.
        x = self.up(x)
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
        pad = 1

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
            indices = (slice(None), slice(None), slice(int(r_d0/scale), int(r_d1/scale)), slice(int(r_h0/scale), int(r_h1/scale)), slice(int(r_w0/scale), int(r_w1/scale)))
            with torch.no_grad():
                patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patchi = conv.chunk_forward(patchi, padlist, indices)
                if onlydown:
                    patchi = pool(patchi)

                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out


    def up_fuse(self, x: Tensor, y: Tensor, conv: BasicBlock, up: UpConv, conv_last=None):
        """
        x = up(x)
        x = torch.cat([y, x], dim=1)
        x = conv(x)
        """
        b, c, d, h, w = x.shape
        pad = 2

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
                # NOTE: We interplote here instead of in up.chunk_forward
                # NOTE: interpolating, then following by padding
                patchi = trilinear_interpolate_align_corners(patchi, (d, h, w), (p_d0, p_h0, p_w0), scale)
                patchi = F.pad(patchi, padlist2, mode='constant', value=0)
                patchi = up.chunk_forward(patchi, padlist2)

                patchj = y[:, :, p_d0*2:p_d1*2, p_h0*2:p_h1*2, p_w0*2:p_w1*2]
                patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                if conv_last is not None:
                    patchj = self.conv1.chunk_forward(patchj, padlist2, None)

                patchi = torch.cat([patchj, patchi], dim=1)
                patchi = conv.chunk_forward(patchi, padlist2)

                if conv_last is not None:
                    patchi = conv_last(patchi)

                out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out