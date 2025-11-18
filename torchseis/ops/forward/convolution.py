# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU).
# All rights reserved.

from typing import Union
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from scipy import ndimage
import math

ArrayLike = Union[np.ndarray, Tensor]


def imp2ref(imp: ArrayLike, taxis: int = -1) -> ArrayLike:
    """
    impedance to reflectivity.

    Parameters
    ----------
    imp : ArrayLike
        Impedance data. It could be 1D (a trace), 2D (2d seismic) or 3D (3d seismic).
    taxis : int, optional
        The time axis along which to compute the reflectivity. Default is -1 (last axis).
    """
    ndim = imp.ndim
    if taxis < 0:
        taxis += ndim
    if taxis < 0 or taxis >= ndim:
        raise ValueError(
            f"Invalid time axis {taxis} for data with {ndim} dimensions.")

    slicer1 = [slice(None)] * ndim
    slicer2 = [slice(None)] * ndim
    slicer1[taxis] = slice(1, None)
    slicer2[taxis] = slice(0, -1)

    if isinstance(imp, torch.Tensor):
        ref = torch.zeros_like(imp, dtype=imp.dtype, device=imp.device)
    else:
        ref = np.zeros_like(imp, dtype=imp.dtype)
    ref[tuple(slicer1)] = (imp[tuple(slicer1)] - imp[tuple(slicer2)]) / (
        imp[tuple(slicer1)] + imp[tuple(slicer2)] + 1e-8)

    return ref


def ref2seis_np(ref: np.ndarray, w: np.ndarray, taxis: int = -1) -> np.ndarray:
    """
    reflecivity to seismic. It's recommended to use a short wavelet

    Parameters
    ----------
    ref : np.ndarray
        Reflectivity data.
    w : np.ndarray
        wavelet, 1D array.
    """
    assert w.ndim == 1, "w must be 1D arrays"
    dtype = ref.dtype
    if dtype == np.float16:
        ref = ref.astype(np.float32)

    seis = ndimage.convolve1d(ref, w, axis=taxis, mode='constant')
    if dtype == np.float16:
        seis = seis.astype(np.float16)
    return seis


def ref2seis_torch(ref: Tensor, w: Tensor, taxis: int = -1) -> Tensor:
    """
    reflecivity to seismic. It's recommended to use a short wavelet

    Parameters
    ----------
    ref : torch.Tensor
        Reflectivity data.
    w : torch.Tensor
        wavelet, 1D array.
    """

    if not (isinstance(ref, torch.Tensor) and isinstance(w, torch.Tensor)):
        raise TypeError("ref and w must both be torch.Tensor")

    ndim = ref.ndim
    if taxis < 0:
        taxis += ndim
    if not (0 <= taxis < ndim):
        raise ValueError(
            f"Invalid time axis {taxis} for data with {ndim} dims.")

    if taxis != ndim - 1:
        perm = list(range(ndim))
        perm.pop(taxis)
        perm.append(taxis)
        ref = ref.permute(perm)  # now shape [..., T]
    else:
        perm = None

    orig_shape = ref.shape
    B = int(torch.prod(torch.tensor(orig_shape[:-1], dtype=torch.long)))
    T = orig_shape[-1]

    ref = ref.reshape(B, T).unsqueeze(1)
    w = w.view(-1)
    w = w.flip(0).view(1, 1,
                       -1)  # flip, as F.conv1d is a correlation operation

    K = w.numel()
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    ref_pad = F.pad(ref, (pad_left, pad_right))
    if w.dtype != ref.dtype:
        w = w.to(ref.dtype)
    if w.device != ref.device:
        w = w.to(ref.device)

    seis = F.conv1d(ref_pad, w)
    seis = seis.squeeze(1).reshape(orig_shape)

    if perm is None:
        return seis

    inv_perm = [0] * ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return seis.permute(inv_perm)


def ref2seis(ref: ArrayLike, w: ArrayLike, taxis: int = -1) -> ArrayLike:
    """
    reflecivity to seismic.

    Parameters
    ----------
    ref : np.ndarray | torch.Tensor
        Reflectivity data.
    w : int
        wavelet, 1D array.
    """
    assert type(ref) == type(w), "ref and w must have the same type"
    if isinstance(ref, np.ndarray):
        return ref2seis_np(ref, w, taxis)
    elif isinstance(ref, torch.Tensor):
        return ref2seis_torch(ref, w, taxis)
    else:
        raise TypeError("ref and w must be either np.ndarray or torch.Tensor")


def ref2seis_batch(ref: Tensor, w: Tensor, taxis: int = -1) -> Tensor:
    """
    Batch version of reflecivity to seismic.

    Parameters
    ----------
    ref : torch.Tensor
        Reflectivity data, shape (B, ...).
    w : torch.Tensor
        wavelet, 1D or 2D array.
    taxis : int, optional
        The time axis along which to compute the reflectivity. Default is -1 (last axis).
    """
    B = ref.shape[0]

    if w.ndim == 1:
        w = w.unsqueeze(0).expand(B, -1)
    elif w.ndim == 2 and w.shape[0] == 1:
        w = w.expand(B, -1)

    assert w.shape[0] == B, "w must have the same batch size as ref"

    if taxis >= 0:
        taxis = taxis - 1
    else:
        taxis = taxis  # negative index, don't need to change

    # we use for loop to avoid consuming too much GPU memory
    # B is usually small, so for loop is acceptable
    for i in range(ref.shape[0]):
        ref[i] = ref2seis_torch(ref[i], w[i], taxis)

    return ref


def ref2seis_nonstat(ref, h, taxis=-1, chunk=0):
    """
    ref: shape = (...)
    h: shape = (..., T, K)

    if chunk <= 0, don't use chunk

    This function doesn't consider `batch` dimension
    """
    # 1) 规范 taxis 维到最后
    ndim = ref.ndim
    if taxis < 0:
        taxis += ndim
    if not (0 <= taxis < ndim):
        raise ValueError(f"Invalid taxis {taxis} for ndim={ndim}")

    if taxis != ndim - 1:
        perm = list(range(ndim))
        perm.pop(taxis)
        perm.append(taxis)
        ref = ref.permute(perm)
    else:
        perm = None

    # 2) 拿出尺寸
    *spatial, T = ref.shape  # spatial = all dims except time
    K = h.size(-1)
    if h.size(-2) != T:
        raise ValueError(
            f"h.shape[-2]={h.size(-2)} must equal time length T={T}")

    # pad 两端保证 same 长度
    pad_l = (K - 1) // 2
    pad_r = (K - 1) - pad_l
    ref = F.pad(ref, (pad_l, pad_r))

    # 先转 (T,K)
    h = h.reshape(-1, h.shape[-2], h.shape[-1])
    # 加 eps 如果需要数值稳定
    # h2 = h2 + eps

    if h.shape[0] == 1:
        if ref.dtype != h.dtype:
            h = h.to(ref.dtype)
        if ref.device != h.device:
            h = h.to(ref.device)

    total = math.prod(spatial)
    ref = ref.reshape(total, -1)

    outs = []
    step = chunk if (chunk and chunk > 0) else total
    for start in range(0, total, step):
        sub = ref[start:start + step]  # (n, T+pad)
        # 滑窗：(n, T, K)
        patches = sub.unfold(-1, K, 1)
        # n = patches.size(0)
        # 构造 kernel：(T,K) -> (n, T, K)
        if h.shape[0] == 1:
            n = patches.size(0)
            ker = h.expand(n, T, K)
        else:
            ker = h[start:start+step]
            if ref.dtype != h.dtype:
                ker = ker.to(ref.dtype)
            if ref.device != h.device:
                ker = ker.to(ref.device)
        # print(ker.shape, patches.shape)
        # print(ker.shape)
        # 点乘并在最后维度求和 -> (n, T)
        outs.append((patches * ker).sum(dim=-1))

    seis = torch.cat(outs, dim=0).reshape(*spatial, T)

    # 4) 如果做过 permute，要恢复维度
    if perm is not None:
        inv = [0] * ndim
        for i, p in enumerate(perm):
            inv[p] = i
        seis = seis.permute(inv)

    return seis


def _get_time(duration: float, dt: float, tc: float, sym: bool = True):
    """
    See bruges's wavelets.py. Here, we add tc to 
    represent the center of the time series
    """
    n = int(duration / dt)
    odd = n % 2
    k = int(10**-np.floor(np.log10(dt)))
    # dti = int(k * dt)  # integer dt

    if (odd and sym):
        t = np.arange(n)
    elif (not odd and sym):
        t = np.arange(n + 1)
    elif (odd and not sym):
        t = np.arange(n)
    elif (not odd and not sym):
        t = np.arange(n) - 1

    t -= t[-1] // 2

    # return dti * t / k + tc
    return t * dt + tc


def ricker(f, dt, l=None, duration=None, sym=True, return_t=False):
    """
    Ricker wavelet: Zero-phase wavelet with a central peak and two smaller
    side lobes. Ricker wavelet is symmetric and centered at 0

    .. math::
        A=\left(1-2 \pi^{2} f^{2} t^{2}\right) e^{-\pi^{2} f^{2} t^{2}}
    """
    if l is not None:
        duration = l * dt
    elif duration is None:
        raise ValueError("Either l or duration must be provided")

    if isinstance(f, (list, tuple)):
        f = np.array(f)
    
    if isinstance(f, np.ndarray):
        f = f[:, np.newaxis]
    
    t = _get_time(duration, dt, 0, sym)[np.newaxis]
    w = (1 - 2 * (np.pi * f * t)**2) * np.exp(-(np.pi * f * t)**2)
    w = np.squeeze(w)

    if return_t:
        return w, t
    else:
        return w
