# Copyright (c) 2025 Jintao Li.
# Zhejiang University (ZJU).
# University of Science and Technology of China (USTC).
# All rights reserved.

import itertools
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .triton_interp3d import trilinear_interpolate_align_corners_triton as trinterp3d


def replace_upsample(module: nn.Module, bsize: int = 32):
    """
    Replace all nn.Upsample in the module with UpsampleChunked
    
    Args:
        module: the module to replace
        bsize: the chunk size, default is 32
    
    Returns:
        the replaced module
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Upsample):
            setattr(module, name, UpsampleChunked(child, bsize))
        elif len(list(child.children())) > 0:
            replace_upsample(child, bsize)

    return module


class UpsampleChunked(nn.Module):

    def __init__(self, ops: nn.Upsample, bsize: int):
        super().__init__()
        self.ops = ops
        self.scale_factor = int(ops.scale_factor)
        assert self.scale_factor - ops.scale_factor == 0
        self.mode = ops.mode
        self.align_corners = ops.align_corners
        self.bsize = bsize

    def forward(self, x: Tensor) -> Tensor:
        return interpolate3d_chunked(
            x,
            self.bsize,
            self.scale_factor,
            self.mode,
            self.align_corners,
        )


def trilinear_interpolate_align_corners(
    input_tensor: Tensor,
    orig_size: tuple[int, int, int],
    start: tuple[int, int, int],
    scale_factor: int = 2,
    force_fp32: bool = False,
) -> Tensor:  # TODO: boost performance
    """
    Note that when using chunked method, the boundary is not correct.
    So we need to handle the boundary manually.
    Besides, compare with F.interpolate, there is noticeable difference (1e-2 for half precision).

    Parameters:
    ------------
    - input_tensor :
        the input tensor, shape: [batch, channel, D_in, H_in, W_in]
    - orig_size: 
        the original size of the input tensor, tuple (D_orig, H_orig, W_orig)
    - start: 
        the start position of the input tensor, tuple (d_start, h_start, w_start)
    - scale_factor: 
        the scale factor

    Returns:
    --------
    - output:
        the output tensor, shape: [batch, channel, D_out, H_out, W_out]
    """
    dtype = input_tensor.dtype
    if force_fp32:
        input_tensor = input_tensor.to(torch.float32)

    batch, channel, D_in, H_in, W_in = input_tensor.shape
    D_out = D_in * scale_factor
    H_out = H_in * scale_factor
    W_out = W_in * scale_factor

    D_or, H_or, W_or = orig_size
    ds, hs, ws = start
    D_outo = D_or * scale_factor
    H_outo = H_or * scale_factor
    W_outo = W_or * scale_factor

    d_os = ds * scale_factor
    h_os = hs * scale_factor
    w_os = ws * scale_factor

    d_oe = d_os + D_out - 1
    h_oe = h_os + H_out - 1
    w_oe = w_os + W_out - 1

    d_out = torch.linspace(d_os, d_oe, D_out, device=input_tensor.device)
    h_out = torch.linspace(h_os, h_oe, H_out, device=input_tensor.device)
    w_out = torch.linspace(w_os, w_oe, W_out, device=input_tensor.device)

    # shift d_out to the position of original data
    d_out = d_out / (D_outo - 1)
    h_out = h_out / (H_outo - 1)
    w_out = w_out / (W_outo - 1)

    # map the output coordinates to the input coordinates (align_corners=True mapping)
    d_in = d_out * (D_or - 1)
    h_in = h_out * (H_or - 1)
    w_in = w_out * (W_or - 1)

    # left/top boundary
    d0 = torch.floor(d_in).long()
    h0 = torch.floor(h_in).long()
    w0 = torch.floor(w_in).long()

    # right/bottom boundary
    d1 = torch.min(d0 + 1, torch.tensor(D_or - 1, device=input_tensor.device))
    h1 = torch.min(h0 + 1, torch.tensor(H_or - 1, device=input_tensor.device))
    w1 = torch.min(w0 + 1, torch.tensor(W_or - 1, device=input_tensor.device))

    # calculate the fractional part
    dd = (d_in - d0.float()).view(1, 1, -1, 1, 1)  # shape [D_out, 1, 1]
    hh = (h_in - h0.float()).view(1, 1, 1, -1, 1)  # shape [1, H_out, 1]
    ww = (w_in - w0.float()).view(1, 1, 1, 1, -1)  # shape [1, 1, W_out]

    # shift d0, h0, w0 d1, h1, w1 to the position of input_tensor
    d0 = d0 - ds
    h0 = h0 - hs
    w0 = w0 - ws
    d1 = d1 - ds
    h1 = h1 - hs
    w1 = w1 - ws

    d0 = torch.clamp(d0, 0, D_in - 1)
    h0 = torch.clamp(h0, 0, H_in - 1)
    w0 = torch.clamp(w0, 0, W_in - 1)
    d1 = torch.clamp(d1, 0, D_in - 1)
    h1 = torch.clamp(h1, 0, H_in - 1)
    w1 = torch.clamp(w1, 0, W_in - 1)

    d0_grid, h0_grid, w0_grid = torch.meshgrid(d0, h0, w0, indexing='ij')
    d1_grid, h1_grid, w1_grid = torch.meshgrid(d1, h1, w1, indexing='ij')

    c000 = input_tensor[:, :, d0_grid, h0_grid, w0_grid]
    c001 = input_tensor[:, :, d0_grid, h0_grid, w1_grid]
    c010 = input_tensor[:, :, d0_grid, h1_grid, w0_grid]
    c011 = input_tensor[:, :, d0_grid, h1_grid, w1_grid]
    c100 = input_tensor[:, :, d1_grid, h0_grid, w0_grid]
    c101 = input_tensor[:, :, d1_grid, h0_grid, w1_grid]
    c110 = input_tensor[:, :, d1_grid, h1_grid, w0_grid]
    c111 = input_tensor[:, :, d1_grid, h1_grid, w1_grid]

    c00 = c000 * (1 - ww) + c001 * ww
    c01 = c010 * (1 - ww) + c011 * ww
    c10 = c100 * (1 - ww) + c101 * ww
    c11 = c110 * (1 - ww) + c111 * ww

    c0 = c00 * (1 - hh) + c01 * hh
    c1 = c10 * (1 - hh) + c11 * hh

    return (c0 * (1 - dd) + c1 * dd).to(dtype)


def interpolate3d_chunked(
    x: Tensor,
    bsize: int,
    scale_factor: int = 2,
    mode: str = 'nearest',
    align_corners: bool | None = None,
    use_triton: bool = True
) -> Tensor:
    r"""
    Chunked 3D upsampling for very large tensors.

    This function splits the input volume into smaller 3D blocks and applies
    local interpolation (``torch.nn.functional.interpolate`` for ``nearest`` or
    a custom trilinear kernel), and then writes back the results into the
    corresponding region of a preallocated output tensor.

    The primary motivation is **not memory saving**, but to **avoid indexing
    overflow and kernel instability** in PyTorch's GPU interpolation kernels
    when processing extremely large 5D tensors (e.g. ``(B, C, D, H, W)`` with
    large spatial sizes). This becomes particularly critical when using **FP16**
    on GPU, where ``F.interpolate`` may produce noticeable errors compared with
    the mathematically correct result.

    Notes
    -----
    * For ``mode='nearest'`` the chunked version tends to be **more numerically
      reliable on GPU FP16** than performing a full-volume
      ``F.interpolate(x, ..., mode='nearest')``. The full-volume version can
      return *incorrect* values (not just precision noise) due to different
      kernel paths or index rounding in half precision.
    * When ``mode='trilinear'`` and ``align_corners=True``, observable
      differences (around ``1e-2`` in FP16) can still occur due to half-precision
      rounding, even though the implementation is mathematically aligned with
      ``F.interpolate``.

    Verification Tip
    ----------------
    To check numerical correctness of the chunked implementation, it is
    recommended to compare against CPU ``float32`` interpolation:

    .. code-block:: python

        x = torch.randn(...).cpu().float()
        out_full = F.interpolate(x, scale_factor=2, mode='nearest')
        out_chunk = interpolate3d_chunked(x, bsize=64, scale_factor=2, mode='nearest')
        print(torch.allclose(out_full, out_chunk, atol=1e-6))

    On GPU FP16, large 3D inputs may make ``F.interpolate`` produce inconsistent
    results, so **CPU results should be treated as the ground truth for testing
    correctness**, not the GPU FP16 version.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(B, C, D, H, W)``.
    bsize : int
        Block size for chunking along spatial dimensions.
    scale_factor : int, default=2
        Upsampling scale factor (currently supports uniform scaling).
    mode : {'nearest', 'trilinear'}
        Interpolation mode.
    align_corners : bool or None
        Same semantics as ``F.interpolate`` (only used for trilinear).
    use_triton : bool
        Whether to use a Triton kernel for trilinear interpolation.

    Returns
    -------
    Tensor
        Upsampled tensor of shape ``(B, C, D*scale_factor, H*scale_factor, W*scale_factor)``.
    """

    assert mode in [
        'nearest', 'trilinear'
    ], f"Only support 'nearest' or 'trilinear' mode, but got {mode}"
    if mode == 'nearest':
        align_corners = None
        pad = 0
    else:
        pad = 1

    b, c, d, h, w = x.shape

    outshape = (
        b,
        c,
        int(d * scale_factor),
        int(h * scale_factor),
        int(w * scale_factor),
    )

    out = torch.zeros(outshape, device=x.device, dtype=x.dtype)

    tsize = bsize - 2 * pad
    nb_d = (d + tsize - 1) // tsize
    nb_h = (h + tsize - 1) // tsize
    nb_w = (w + tsize - 1) // tsize

    for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)):
        v_d0 = i * tsize
        v_h0 = j * tsize
        v_w0 = k * tsize

        v_d1 = min(v_d0 + tsize, d)
        v_h1 = min(v_h0 + tsize, h)
        v_w1 = min(v_w0 + tsize, w)

        p_d0 = max(0, v_d0 - pad)
        p_h0 = max(0, v_h0 - pad)
        p_w0 = max(0, v_w0 - pad)

        p_d1 = min(d, v_d1 + pad)
        p_h1 = min(h, v_h1 + pad)
        p_w1 = min(w, v_w1 + pad)

        r_d0 = (v_d0 - p_d0) * scale_factor
        r_d1 = r_d0 + (v_d1 - v_d0) * scale_factor
        r_h0 = (v_h0 - p_h0) * scale_factor
        r_h1 = r_h0 + (v_h1 - v_h0) * scale_factor
        r_w0 = (v_w0 - p_w0) * scale_factor
        r_w1 = r_w0 + (v_w1 - v_w0) * scale_factor

        o_d0 = v_d0 * scale_factor
        o_h0 = v_h0 * scale_factor
        o_w0 = v_w0 * scale_factor

        o_d1 = v_d1 * scale_factor
        o_h1 = v_h1 * scale_factor
        o_w1 = v_w1 * scale_factor

        with torch.no_grad():
            patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
            if patchi.numel() == 0:
                continue
            if align_corners:
                if use_triton:
                    patcho = trinterp3d(
                        patchi,
                        (d, h, w),
                        (p_d0, p_h0, p_w0),
                        scale_factor,
                    )
                else:
                    patcho = trilinear_interpolate_align_corners(
                        patchi,
                        (d, h, w),
                        (p_d0, p_h0, p_w0),
                        scale_factor,
                    )
            else:
                patcho = F.interpolate(
                    patchi,
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corners,
                )
            out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1] # yapf: disable

    return out
