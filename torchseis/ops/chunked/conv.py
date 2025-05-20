# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import itertools
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.ao.nn.intrinsic.modules.fused import ConvReLU3d


def set_pad_as_zero(x, padlist):
    """
    x is from `F.pad` with padlist,set the padding area as zero
    """
    if padlist[0] > 0:
        x[:, :, :, :, :padlist[0]] = 0
    if padlist[1] > 0:
        x[:, :, :, :, -padlist[1]:] = 0
    if padlist[2] > 0:
        x[:, :, :, :padlist[2], :] = 0
    if padlist[3] > 0:
        x[:, :, :, -padlist[3]:, :] = 0
    if padlist[4] > 0:
        x[:, :, :padlist[4], :, :] = 0
    if padlist[5] > 0:
        x[:, :, -padlist[5]:, :, :] = 0
    return x


def replace_conv(module: nn.Module, bsize: int = 128):
    """
    Replace all Conv3d or ConvReLU3d in the module with Conv3dChunked
    
    Args:
        module: 要替换的模块
        bsize: 分块大小，默认为128
    
    Returns:
        替换后的模块
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            setattr(module, name, Conv3dChunked(child, bsize))
        elif isinstance(child, ConvReLU3d):
            setattr(module, name, Conv3dChunked(child, bsize))
        elif isinstance(child, nn.ConvTranspose3d):
            setattr(module, name, ConvTranspose3dChunked(child, bsize))
        elif len(list(child.children())) > 0:
            # 递归处理子模块
            replace_conv(child, bsize)

    return module


class Conv3dChunked(nn.Module):

    def __init__(self, ops: nn.Conv3d, bsize: int):
        super().__init__()
        self.ops = ops
        self.bsize = bsize

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return conv3d_chunked(x, self.ops, self.bsize, **kwargs)


class ConvTranspose3dChunked(nn.Module):

    def __init__(self, ops: nn.ConvTranspose3d, bsize: int):
        super().__init__()
        self.ops = ops
        self.bsize = bsize

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return conv_transpose3d_chunked(x, self.ops, self.bsize, **kwargs)


def conv3d_chunked(
    x: Tensor,
    ops: nn.Conv3d,
    bsize: int,
    **kwargs,
) -> Tensor:
    assert isinstance(ops, (nn.Conv3d, ConvReLU3d)), "Only support Conv3d"
    assert x.ndim == 5, "Only support 5D input"

    b, c, d, h, w = x.shape
    if isinstance(ops, ConvReLU3d):
        out_c = ops[0].out_channels
        pad = ops[0].padding
        padding_mode = ops[0].padding_mode
        ops[0].padding = 0
        dilation = ops[0].dilation
        stride = ops[0].stride
        kernel_size = ops[0].kernel_size
    else:
        out_c = ops.out_channels
        pad = ops.padding
        padding_mode = ops.padding_mode
        ops.padding = 0
        dilation = ops.dilation
        stride = ops.stride
        kernel_size = ops.kernel_size

    if padding_mode == "zeros":
        padding_mode = "constant"
        padding_value = 0
    else:
        padding_value = None

    # fmt: off
    out_d = (d + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_h = (h + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    out_w = (w + 2 * pad[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1

    assert d % out_d == 0 and h % out_h == 0 and w % out_w == 0, "Only support input and output have a multiple relationship"
    scale = (d // out_d, h // out_h, w // out_w)

    assert bsize % stride[0] == 0 and bsize % stride[1] == 0 and bsize % stride[2] == 0, "Only support input and output have a multiple relationship"
    # fmt: on

    # tsize = tuple(bsize - 2 * p for p in pad)
    tsize = (bsize, bsize, bsize)

    nb_d = (d + tsize[0] - 1) // tsize[0]
    nb_h = (h + tsize[1] - 1) // tsize[1]
    nb_w = (w + tsize[2] - 1) // tsize[2]

    out = torch.zeros(
        (b, out_c, out_d, out_h, out_w),
        device=x.device,
        dtype=x.dtype,
    )

    for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
        v_d0 = i * tsize[0]
        v_h0 = j * tsize[1]
        v_w0 = k * tsize[2]

        v_d1 = min(v_d0 + tsize[0], d)
        v_h1 = min(v_h0 + tsize[1], h)
        v_w1 = min(v_w0 + tsize[2], w)

        p_d0 = max(0, v_d0 - pad[0])
        p_h0 = max(0, v_h0 - pad[1])
        p_w0 = max(0, v_w0 - pad[2])

        p_d1 = min(d, v_d1 + pad[0])
        p_h1 = min(h, v_h1 + pad[1])
        p_w1 = min(w, v_w1 + pad[2])

        padlist = [0, 0, 0, 0, 0, 0]

        if v_d1 == p_d1:
            padlist[5] = pad[0]
        if v_h1 == p_h1:
            padlist[3] = pad[1]
        if v_w1 == p_w1:
            padlist[1] = pad[2]
        if v_d0 == p_d0:
            padlist[4] = pad[0]
        if v_h0 == p_h0:
            padlist[2] = pad[1]
        if v_w0 == p_w0:
            padlist[0] = pad[2]

        r_d0 = int((v_d0 - p_d0 + padlist[4]) * scale[0])
        r_d1 = r_d0 + int((v_d1 - v_d0) * scale[0])
        r_h0 = int((v_h0 - p_h0 + padlist[2]) * scale[1])
        r_h1 = r_h0 + int((v_h1 - v_h0) * scale[1])
        r_w0 = int((v_w0 - p_w0 + padlist[0]) * scale[2])
        r_w1 = r_w0 + int((v_w1 - v_w0) * scale[2])

        o_d0 = v_d0 // scale[0]
        o_h0 = v_h0 // scale[1]
        o_w0 = v_w0 // scale[2]

        o_d1 = v_d1 // scale[0]
        o_h1 = v_h1 // scale[1]
        o_w1 = v_w1 // scale[2]

        with torch.no_grad():
            patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
            if sum(padlist) > 0:
                patcho = F.pad(
                    patchi,
                    padlist,
                    mode=padding_mode,
                    value=padding_value,
                )
            else:
                patcho = patchi
            patcho = ops(patcho, **kwargs)
            # patcho = ops(patcho, **kwargs)[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

            out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho

    if isinstance(ops, ConvReLU3d):
        ops[0].padding = pad
    else:
        ops.padding = pad
    return out


def conv_transpose3d_chunked(
    x: Tensor,
    ops: nn.ConvTranspose3d,
    bsize: int,
    **kwargs,
) -> Tensor:
    """
    If the padding is enabled in ops, there is noticeable error when using torch.float32.
    I don't know why.
    """
    assert isinstance(ops, nn.ConvTranspose3d), "Only support ConvTranspose3d"
    assert x.ndim == 5, "Only support 5D input"

    b, c, d, h, w = x.shape
    out_c = ops.out_channels
    pad = ops.padding

    # fmt: off
    out_d = (d - 1) * ops.stride[0] - 2 * pad[0] + ops.dilation[0] * (ops.kernel_size[0] - 1) + ops.output_padding[0] + 1
    out_h = (h - 1) * ops.stride[1] - 2 * pad[1] + ops.dilation[1] * (ops.kernel_size[1] - 1) + ops.output_padding[1] + 1
    out_w = (w - 1) * ops.stride[2] - 2 * pad[2] + ops.dilation[2] * (ops.kernel_size[2] - 1) + ops.output_padding[2] + 1

    assert out_d % d == 0 and out_h % h == 0 and out_w % w == 0, "Only support input and output have a multiple relationship"
    scale = (out_d // d, out_h // h, out_w // w)

    assert bsize % ops.stride[0] == 0 and bsize % ops.stride[1] == 0 and bsize % ops.stride[2] == 0, "Only support input and output have a multiple relationship"
    # fmt: on

    tsize = tuple(bsize - 2 * p for p in pad)

    nb_d = (d + tsize[0] - 1) // tsize[0]
    nb_h = (h + tsize[1] - 1) // tsize[1]
    nb_w = (w + tsize[2] - 1) // tsize[2]

    out = torch.zeros(
        (b, out_c, out_d, out_h, out_w),
        device=x.device,
        dtype=x.dtype,
    )

    for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
        v_d0 = i * tsize[0]
        v_h0 = j * tsize[1]
        v_w0 = k * tsize[2]

        v_d1 = min(v_d0 + tsize[0], d)
        v_h1 = min(v_h0 + tsize[1], h)
        v_w1 = min(v_w0 + tsize[2], w)

        p_d0 = max(0, v_d0 - pad[0])
        p_h0 = max(0, v_h0 - pad[1])
        p_w0 = max(0, v_w0 - pad[2])

        p_d1 = min(d, v_d1 + pad[0])
        p_h1 = min(h, v_h1 + pad[1])
        p_w1 = min(w, v_w1 + pad[2])

        r_d0 = (v_d0 - p_d0) * scale[0]
        r_d1 = r_d0 + (v_d1 - v_d0) * scale[0]
        r_h0 = (v_h0 - p_h0) * scale[1]
        r_h1 = r_h0 + (v_h1 - v_h0) * scale[1]
        r_w0 = (v_w0 - p_w0) * scale[2]
        r_w1 = r_w0 + (v_w1 - v_w0) * scale[2]

        o_d0 = v_d0 * scale[0]
        o_h0 = v_h0 * scale[1]
        o_w0 = v_w0 * scale[2]

        o_d1 = v_d1 * scale[0]
        o_h1 = v_h1 * scale[1]
        o_w1 = v_w1 * scale[2]

        with torch.no_grad():
            patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
            patcho = ops(patchi, **kwargs)[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1] # yapf: disable
            out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho

    return out