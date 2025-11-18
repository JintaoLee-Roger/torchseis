import itertools
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...ops.chunked.conv import set_pad_as_zero, Conv3dChunked
from ...ops.chunked.norms import AdaptiveInstanceNorm3d, instance_norm
from ...ops.fuse import ConvInstanceNorm3d
from ...ops.chunked.triton_interp3d import trilinear_interpolate_align_corners_triton as trinterp3d
from ...ops.chunked.statistics import ChunkedAdaptiveAvgPool3d1
import time


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

    def chunked_forward(self, x: Tensor, t: int = -1, indices=None) -> Tensor:
        b, c, _, _, _ = x.size()
        if t == 0:
            self.avg_pool.add_batch(x[indices])
            return
        elif not self.avg_pool.has_statistics():
            self.avg_pool.compute()
        y = self.avg_pool.out.view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

    def replace_norm(self):
        if isinstance(self.avg_pool, ChunkedAdaptiveAvgPool3d1):
            self.avg_pool.reset()
        else:
            self.avg_pool = ChunkedAdaptiveAvgPool3d1()

    def reset(self):
        if isinstance(self.avg_pool, ChunkedAdaptiveAvgPool3d1):
            self.avg_pool.reset()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=(1, 1),
                 residual=True,
                 norm=None,
                 se=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation[0],
                               bias=False,
                               dilation=dilation[0])
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=dilation[1],
                               bias=False,
                               dilation=dilation[1])
        self.bn2 = norm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual
        if se:
            self.se = SELayer(planes)
        else:
            self.se = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        if self.se is not None:
            out = self.relu(out)
        return out


class Attention_block(nn.Module):

    def __init__(self, F_g, F_x, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.InstanceNorm3d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv3d(F_x,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True), nn.InstanceNorm3d(F_int))

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.fused = False

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

    def chunk_conv_forward(self, g, x):
        g1 = Conv3dChunked(self.W_g[0], 64)(g)
        g1 = self.W_g[1](g1)
        x1 = Conv3dChunked(self.W_x[0], 64)(x)
        x1 = self.W_x[1](x1)
        psi = self.relu(g1 + x1)
        psi = Conv3dChunked(self.psi[0], 64)(psi)
        psi = self.psi[1:](psi)
        out = x * psi
        return out

    def chunked_forward(self, g, x, padlist, t=-1, indices=None):
        g1 = self.W_g[0](g)
        x1 = self.W_x[0](x)

        if t == 0:
            self.W_g[1].stats.add_batch(g1[indices])
            self.W_x[1].stats.add_batch(x1[indices])
            return
        elif not self.W_g[1].stats.has_statistics():
            self.W_g[1].stats.compute()
            self.W_x[1].stats.compute()

        g1 = self.W_g[1](g1)
        x1 = self.W_x[1](x1)
        g1 = set_pad_as_zero(g1, padlist)
        x1 = set_pad_as_zero(x1, padlist)
        psi = self.relu(g1 + x1)

        psi = self.psi[0](psi)
        if t == 1:
            self.psi[1].stats.add_batch(psi[indices])
            return psi
        elif not self.psi[1].stats.has_statistics():
            self.psi[1].stats.compute()

        psi = self.psi[1:](psi)
        psi = set_pad_as_zero(psi, padlist)
        out = x * psi
        return out

    def replace_norm(self):
        if isinstance(self.W_g[1], AdaptiveInstanceNorm3d):
            self.W_g[1].stats.reset()
            self.W_x[1].stats.reset()
            self.psi[1].stats.reset()
        else:
            self.W_g[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.W_g[1])
            self.W_x[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.W_x[1])
            self.psi[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.psi[1])

    def reset(self):
        if isinstance(self.W_g[1], AdaptiveInstanceNorm3d):
            self.W_g[1].stats.reset()
            self.W_x[1].stats.reset()
            self.psi[1].stats.reset()


class UP(nn.Sequential):

    def __init__(self, in_c_g, in_c_x, out_c):
        super(UP, self).__init__()
        # attention block
        self.out_c = out_c
        self.atte = Attention_block(in_c_g, in_c_x, in_c_x)
        num_input_features = in_c_g + in_c_x

        # conventional conv
        self.conv = nn.Conv3d(num_input_features,
                              out_c,
                              kernel_size=5,
                              stride=1,
                              padding=2,
                              bias=False)
        self.bn = nn.InstanceNorm3d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(out_c)

    def forward(self, x, x1, size):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)

        if self.atte is not None:
            x1 = self.atte(x, x1)
        x = torch.cat([x, x1], dim=1)

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.se is not None:
            out = self.se(out)
        return out

    def chunk_conv_forward(self, x, x1, size):
        x = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        if self.atte is not None:
            x1 = self.atte.chunk_conv_forward(x, x1)
        x = torch.cat([x, x1], dim=1)
        out = Conv3dChunked(self.conv, 64)(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.se is not None:
            out = self.se(out)
        return out

    def fusion(self, x, x1, bsize=64):
        b, c, d, h, w = x1.shape
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        outc = self.out_c
        pad = 2  # full resolution is 4, but half resolution is 2
        oshape = (b, outc, d, h, w)
        scale = 2

        self.atte.replace_norm()

        out = torch.zeros(oshape, dtype=x.dtype, device=x.device)
        for t in range(3):
            for i, j, k in itertools.product(range(nb_d), range(nb_h),
                                             range(nb_w)):
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = _get_index(
                    i, j, k, bsize, (d // 2, h // 2, w // 2), pad, scale)
                padlist2 = [f * 2 for f in padlist]
                indices1 = (slice(None), slice(None),
                            slice(r_d0 // 2,
                                  r_d1 // 2), slice(r_h0 // 2, r_h1 // 2),
                            slice(r_w0 // 2, r_w1 // 2))
                indices2 = (slice(None), slice(None), slice(r_d0, r_d1),
                            slice(r_h0, r_h1), slice(r_w0, r_w1))

                with torch.no_grad():
                    patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    if patchi.numel() == 0:
                        continue
                    # NOTE: We interplote here instead of in up.chunk_forward
                    # NOTE: interpolating, then following by padding
                    # patchi = trilinear_interpolate_align_corners(patchi, x.shape[2:], (p_d0, p_h0, p_w0), scale_factor=2)
                    patchi = trinterp3d(patchi,
                                        x.shape[2:], (p_d0, p_h0, p_w0),
                                        scale_factor=2)
                    patchi = F.pad(patchi, padlist2, mode='constant', value=0)

                    patchj = x1[:, :, p_d0 * 2:p_d1 * 2, p_h0 * 2:p_h1 * 2,
                                p_w0 * 2:p_w1 * 2]
                    patchj = F.pad(patchj, padlist2, mode='constant', value=0)

                    # attention
                    patchj = self.atte.chunked_forward(patchi, patchj,
                                                       padlist2, t, indices2)
                    if t < 2:
                        continue

                    patchi = torch.cat([patchi, patchj], dim=1)

                    patchi = self.conv(patchi)  # need pad 2

                    out[:, :, o_d0:o_d1, o_h0:o_h1,
                        o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1,
                                            r_w0:r_w1]
        return out


class _FuseOps:

    def _layer012(self, x: Tensor, bsize: int = 64):
        if not isinstance(self.layer0[1], AdaptiveInstanceNorm3d):
            self.layer0[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.layer0[1])
        if not isinstance(self.layer1[1], AdaptiveInstanceNorm3d):
            self.layer1[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.layer1[1])
        if not isinstance(self.layer2[1], AdaptiveInstanceNorm3d):
            self.layer2[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.layer2[1])

        b, c, d, h, w = x.shape
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        outc = 32
        pad = 6
        oshape = (b, outc, d // 2, h // 2, w // 2)
        scale = 0.5

        out = torch.zeros(oshape, dtype=x.dtype, device=x.device)
        for t in range(4):
            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = _get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable
                indices1 = (slice(None), slice(None),
                            slice(r_d0 * 2,
                                  r_d1 * 2), slice(r_h0 * 2, r_h1 * 2),
                            slice(r_w0 * 2, r_w1 * 2))
                indices2 = (slice(None), slice(None), slice(r_d0, r_d1),
                            slice(r_h0, r_h1), slice(r_w0, r_w1))

                # indices1 = (slice(None), slice(None), slice(r_d0, r_d1), slice(r_h0, r_h1), slice(r_w0, r_w1))
                # indices2 = None

                with torch.no_grad():
                    patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    patchi = F.pad(patchi, padlist, mode='constant', value=0)
                    patcho = self.layer0[0](patchi)  # need pad 3
                    if t == 0:
                        self.layer0[1].stats.add_batch(patcho[indices1])
                        continue
                    elif not self.layer0[1].stats.has_statistics():
                        self.layer0[1].stats.compute()
                        if self.fused:
                            mean = self.layer0[1].stats.mean
                            var = self.layer0[1].stats.var
                            self.layer0[0] = ConvInstanceNorm3d(
                                self.layer0[0], self.layer0[1], mean, var)
                            self.layer0[1].fused = True
                    patcho = self.layer0[1:](patcho)
                    patcho = set_pad_as_zero(patcho, padlist)

                    patcho = self.layer1[0](patcho)  # need pad 1
                    if t == 1:
                        self.layer1[1].stats.add_batch(patcho[indices1])
                        continue
                    elif not self.layer1[1].stats.has_statistics():
                        self.layer1[1].stats.compute()
                        if self.fused:
                            mean = self.input_layer2[1].stats.mean
                            var = self.input_layer2[1].stats.var
                            self.input_layer2[0] = ConvInstanceNorm3d(
                                self.input_layer2[0], self.input_layer2[1],
                                mean, var)
                            self.input_layer2[1].fused = True
                    patcho = self.layer1[1:](patcho)
                    patcho = set_pad_as_zero(patcho, padlist)

                    patcho = self.layer2[0](patcho)  # need pad 2
                    if t == 2:
                        self.layer2[1].stats.add_batch(patcho[indices2])
                        continue
                    elif not self.layer2[1].stats.has_statistics():
                        self.layer2[1].stats.compute()
                        if self.fused:
                            mean = self.input_layer3[1].stats.mean
                            var = self.input_layer3[1].stats.var
                            self.input_layer3[0] = ConvInstanceNorm3d(
                                self.input_layer3[0], self.input_layer3[1],
                                mean, var)
                            self.input_layer3[1].fused = True
                    patcho = self.layer2[1:](patcho)

                    out[:, :, o_d0:o_d1, o_h0:o_h1,
                        o_w0:o_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1,
                                            r_w0:r_w1]

        return out

    def _up5(self, up4: Tensor, x: Tensor, bsize=64):
        b, c, d, h, w = x.shape
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        outc = 1
        pad = 3  # full resolution is 6, but half resolution is 3
        oshape = (b, outc, d, h, w)
        scale = 2

        self.up5.atte.replace_norm()
        self.up5.se.replace_norm()
        if not isinstance(self.up5.bn, AdaptiveInstanceNorm3d):
            self.up5.bn = AdaptiveInstanceNorm3d.from_instance_norm(self.up5.bn)
        else:
            self.up5.bn.stats.reset()
            self.up5.bn.fused = False

        out = torch.zeros(oshape, dtype=x.dtype, device=x.device)
        for t in range(5):
            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = _get_index(i, j, k, bsize, (d // 2, h // 2, w // 2), pad, scale)
                padlist2 = [f * 2 for f in padlist]
                # indices1 = (slice(None), slice(None), slice(r_d0 // 2, r_d1 // 2), slice(r_h0 // 2, r_h1 // 2), slice(r_w0 // 2, r_w1 // 2))
                indices2 = (slice(None), slice(None), slice(r_d0, r_d1), slice(r_h0, r_h1), slice(r_w0, r_w1))

                with torch.no_grad():
                    patchi = up4[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    # NOTE: We interplote here instead of in up.chunk_forward
                    # NOTE: interpolating, then following by padding
                    if patchi.numel() == 0:
                        continue
                    # patchi = F.interpolate(patchi, scale_factor=2, mode='trilinear', align_corners=True)
                    # patchi = trilinear_interpolate_align_corners(patchi, up4.shape[2:], (p_d0, p_h0, p_w0), scale_factor=2)
                    patchi = trinterp3d(patchi, up4.shape[2:], (p_d0, p_h0, p_w0), scale_factor=2)
                    patchi = F.pad(patchi, padlist2, mode='constant', value=0)

                    patchj = x[:, :, p_d0 * 2:p_d1 * 2, p_h0 * 2:p_h1 * 2, p_w0 * 2:p_w1 * 2]
                    patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                    patchj = self.layer0(patchj) # need pad 3
                    patchj = set_pad_as_zero(patchj, padlist2)

                    # attention
                    patchj = self.up5.atte.chunked_forward(patchi, patchj, padlist2, t, indices2)
                    if t < 2:
                        continue

                    patchi = torch.cat([patchi, patchj], dim=1)

                    patchi = self.up5.conv(patchi) # need pad 2
                    if t == 2:
                        self.up5.bn.stats.add_batch(patchi[indices2])
                        continue
                    elif not self.up5.bn.stats.has_statistics():
                        self.up5.bn.stats.compute()
                        if self.fused:
                            mean = self.up5.bn.stats.mean
                            var = self.up5.bn.stats.var
                            self.up5.conv = ConvInstanceNorm3d(self.up5.conv, self.up5.bn, mean, var)
                            self.up5.bn.fused = True
                    patchi = self.up5.bn(patchi)
                    patchi = self.up5.relu(patchi)
                    patchi = set_pad_as_zero(patchi, padlist2)

                    # se
                    patchi = self.up5.se.chunked_forward(patchi, t - 3, indices2)
                    if t < 4:
                        continue
                    patchi = self.out_layer[0](patchi)

                    out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patchi[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out


def _get_index(i, j, k, bsize, shape, pad, scale=1):
    v_d0 = i * bsize
    v_h0 = j * bsize
    v_w0 = k * bsize
    d, h, w = shape

    v_d1 = min(v_d0 + bsize, d)
    v_h1 = min(v_h0 + bsize, h)
    v_w1 = min(v_w0 + bsize, w)

    p_d0 = max(0, v_d0 - pad)
    p_h0 = max(0, v_h0 - pad)
    p_w0 = max(0, v_w0 - pad)

    p_d1 = min(d, v_d1 + pad)
    p_h1 = min(h, v_h1 + pad)
    p_w1 = min(w, v_w1 + pad)

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

    r_d0 = int((v_d0 - p_d0 + padlist[4]) * scale)
    r_d1 = r_d0 + int((v_d1 - v_d0) * scale)
    r_h0 = int((v_h0 - p_h0 + padlist[2]) * scale)
    r_h1 = r_h0 + int((v_h1 - v_h0) * scale)
    r_w0 = int((v_w0 - p_w0 + padlist[0]) * scale)
    r_w1 = r_w0 + int((v_w1 - v_w0) * scale)

    o_d0 = int(v_d0 * scale)
    o_h0 = int(v_h0 * scale)
    o_w0 = int(v_w0 * scale)

    o_d1 = int(v_d1 * scale)
    o_h1 = int(v_h1 * scale)
    o_w1 = int(v_w1 * scale)

    return v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist
