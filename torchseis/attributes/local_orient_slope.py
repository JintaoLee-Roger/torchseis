# Copyright (c) 2025 Jintao Li. 
# Zhejiang University (ZJU), and 
# University of Science and Technology of China (USTC).
# All rights reserved.
# Email: lijintao@mail.ustc.edu.cn

"""
Calculate the Orientations of seismic images using Structure Tensors:
A python (torch) implementation follows the java code's:
https://github.com/xinwucwp/mhe/blob/a458d9bd5185cde0e03fc95dd4df3f91174f4872/src/mhe/demo3.py#L39
"""

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F
import math

__all__ = ['structure_tensor_orientations2d', 'structure_tensor_orientations3d']

def gaussian_kernel1d(
    sigma: float,
    truncate: float = 3.0,
    device=None,
    dtype=None,
) -> Tensor:
    if sigma <= 0:  # identity
        k = torch.ones(1, device=device, dtype=dtype)
        return k / k.sum()
    r = int(math.ceil(truncate * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma)**2)
    return k / k.sum()


def _pad_t(ksize, dim, ndim=3):
    pad = [0] * ndim * 2
    pad[2 * (ndim - dim - 1)] = (ksize - 1) // 2
    pad[2 * (ndim - dim - 1) + 1] = (ksize - 1) // 2
    return tuple(pad)


def gaussian_smooth2d(
    x: Tensor,
    sigma: Union[float, List[float]],
    truncate: float = 3.0,
) -> Tensor:
    """
    2D Gaussian smoothing
    """
    # x: (B,C,H,W)
    B, C, H, W = x.shape
    dev, dty = x.device, x.dtype
    if isinstance(sigma, (float, int)):
        sigma = [sigma, sigma]
    sigma1, sigma2 = sigma
    if sigma1 > 0:
        kh = gaussian_kernel1d(sigma1, truncate, dev, dty).view(1, 1, -1, 1)
        x = F.pad(x, _pad_t(kh.shape[2], 0, 2), mode='reflect')
        x = F.conv2d(x, kh.expand(C, 1, -1, 1), groups=C)
    if sigma2 > 0:
        kw = gaussian_kernel1d(sigma2, truncate, dev, dty).view(1, 1, 1, -1)
        x = F.pad(x, _pad_t(kw.shape[3], 1, 2), mode='reflect')
        x = F.conv2d(x, kw.expand(C, 1, 1, -1), groups=C)
    return x


def gaussian_smooth3d(
    x: Tensor,
    sigma: Union[float, List[float]],
    truncate: float = 3.0,
    pad_mode: str = 'reflect',
) -> Tensor:
    """
    x: (B,C,T,Y,X)
    """
    B, C, T, Y, X = x.shape
    dev, dty = x.device, x.dtype
    if isinstance(sigma, (float, int)):
        sigma = [sigma, sigma, sigma]
    sigma1, sigma2, sigma3 = sigma

    if sigma1 > 0:
        kt = gaussian_kernel1d(sigma1, truncate, dev, dty).view(1, 1, -1, 1, 1)
        x = F.pad(x, _pad_t(kt.shape[2], 0, 3), mode=pad_mode)
        x = F.conv3d(x, kt.expand(C, 1, -1, 1, 1), groups=C)

    if sigma2 > 0:
        ky = gaussian_kernel1d(sigma2, truncate, dev, dty).view(1, 1, 1, -1, 1)
        x = F.pad(x, _pad_t(ky.shape[3], 1, 3), mode=pad_mode)
        x = F.conv3d(x, ky.expand(C, 1, 1, -1, 1), groups=C)

    if sigma3 > 0:
        kx = gaussian_kernel1d(sigma3, truncate, dev, dty).view(1, 1, 1, 1, -1)
        x = F.pad(x, _pad_t(kx.shape[4], 2, 3), mode=pad_mode)
        x = F.conv3d(x, kx.expand(C, 1, 1, 1, -1), groups=C)

    return x


def central_diff2d(x: Tensor, h: Union[float, List[float]] = 1.0) -> Tensor:
    # x: (B,1,H,W), return gt=∂/∂t(vertical axis H), gx=∂/∂x(Horizontal axis W)
    if isinstance(h, (float, int)):
        h = [h, h]
    dt, dx = h
    dev, dty = x.device, x.dtype
    kt = torch.tensor([-0.5, 0, 0.5], device=dev, dtype=dty).view(1, 1, 3, 1) / dt # yapf: disable
    kx = torch.tensor([-0.5, 0, 0.5], device=dev, dtype=dty).view(1, 1, 1, 3) / dx # yapf: disable
    gt = F.conv2d(F.pad(x, (0, 0, 1, 1), mode='reflect'), kt)
    gx = F.conv2d(F.pad(x, (1, 1, 0, 0), mode='reflect'), kx)
    return gt, gx


def central_diff3d(
    x: Tensor,
    h: Union[float, List[float]]=1.0,
    pad_mode: str = 'reflect',
) -> Tensor:
    """
    x: (B,1,T,Y,X) -> gt, gy, gx
    """
    if isinstance(h, (float, int)):
        h = [h, h, h]
    dt, dy, dx = h
    dev, dty = x.device, x.dtype
    kt = torch.tensor([-0.5, 0.0, 0.5], device=dev, dtype=dty).view(1, 1, 3, 1, 1) / dt # yapf: disable
    ky = torch.tensor([-0.5, 0.0, 0.5], device=dev, dtype=dty).view(1, 1, 1, 3, 1) / dy # yapf: disable
    kx = torch.tensor([-0.5, 0.0, 0.5], device=dev, dtype=dty).view(1, 1, 1, 1, 3) / dx # yapf: disable

    gt = F.conv3d(F.pad(x, (0, 0, 0, 0, 1, 1), mode=pad_mode), kt)
    gy = F.conv3d(F.pad(x, (0, 0, 1, 1, 0, 0), mode=pad_mode), ky)
    gx = F.conv3d(F.pad(x, (1, 1, 0, 0, 0, 0), mode=pad_mode), kx)
    return gt, gy, gx



def structure_tensor_orientations2d(
    x: Tensor,
    taxis: int = -1,
    sigma: Union[float, List[float]] = 2.0,
    outputs: Iterable[Literal["u","eigs","v","theta","lty"]] = ("u",),
    enforce: bool = True,
) -> Dict:
    """
    outputs should contain:
      - "u"   : List[Tensor], normal vector, 1st (u1) and 2nd (u2) components of 1st eigenvector. [u1, u2]
      - "eigs": List[Tensor], eigenvalues corresponding to the eigenvectors u and v. [eu, ev] and eu >= ev
      - "v"   : List[Tensor], orthogonal basis。[v1, v2]
      - "theta": Tensor, angles, arcsin(u2)
      - "lty": Tensor, linearity, (eu - ev) / eu
    """
    eps = 1e-7
    assert x.ndim == 4 and x.shape[1] == 1
    if taxis == -1 or taxis == 3:
        gx, gt = central_diff2d(x)  # HACK: We ignore dt and dx
        if isinstance(sigma, list or tuple):
            sigma = list(sigma)[::-1]
    else:
        gt, gx = central_diff2d(x)

    gtt = gaussian_smooth2d(gt * gt, sigma).squeeze(1)  # (B,H,W)
    gxx = gaussian_smooth2d(gx * gx, sigma).squeeze(1)
    gtx = gaussian_smooth2d(gt * gx, sigma).squeeze(1)

    tr = gtt + gxx
    det = gtt * gxx - gtx * gtx
    tmp = torch.sqrt(torch.clamp(tr * tr * 0.25 - det, min=0.0))
    eu = tr * 0.5 + tmp
    ev = tr * 0.5 - tmp

    # Principal feature vector, degradation protection
    v0 = eu - gxx
    v1 = gtx
    nrm = torch.sqrt(torch.clamp(v0 * v0 + v1 * v1, min=eps))
    v0 = v0 / nrm
    v1 = v1 / nrm

    u1 = v0.unsqueeze(1)
    u2 = v1.unsqueeze(1)
    eu = eu.unsqueeze(1)
    ev = ev.unsqueeze(1)

    if enforce:
        s = torch.where(u1 < 0, -1.0, 1.0)
        u1 = u1 * s
        u2 = u2 * s

    if taxis == -1 or taxis == 3:
        u2, u1 = u1, u2

    out = {}
    if "u" in outputs:
        out["u"] = [u1, u2]
    if "v" in outputs:
        out["v"] = [u2, -u1]
    if "eigs" in outputs:
        out["eigs"] = [eu, ev]
    if "theta" in outputs:
        out['theta'] = torch.arcsin(u2)
    if "lty" in outputs:
        out["lty"] = (eu - ev) / torch.clamp(eu, min=eps)


    # # 在严重退化区(例如 eu≈ev≈0)避免全0：可选把向量设为(1,0)
    # mask = (el < 1e-3) | (eu < 1e-8)
    # if mask.any():
    #     u1 = torch.where(mask, torch.ones_like(u1), u1)
    #     u2 = torch.where(mask, torch.zeros_like(u2), u2)

    return out


def _smoothed_products(
    gt: Tensor,
    gy: Tensor,
    gx: Tensor,
    sigma: Union[float, List[float]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    gtt = gt * gt
    gyy = gy * gy
    gxx = gx * gx
    gty = gt * gy
    gtx = gt * gx
    gyx = gy * gx
    gtt = gaussian_smooth3d(gtt, sigma).reshape(-1)
    gyy = gaussian_smooth3d(gyy, sigma).reshape(-1)
    gxx = gaussian_smooth3d(gxx, sigma).reshape(-1)
    gty = gaussian_smooth3d(gty, sigma).reshape(-1)
    gtx = gaussian_smooth3d(gtx, sigma).reshape(-1)
    gyx = gaussian_smooth3d(gyx, sigma).reshape(-1)
    return gtt, gyy, gxx, gty, gtx, gyx


def _chunk_eigh(
    gtt: Tensor,
    gyy: Tensor,
    gxx: Tensor,
    gty: Tensor,
    gtx: Tensor,
    gyx: Tensor,
    shape: Tuple,
    outputs: Iterable[Literal["u","v","w","eigs","theta","phi","lty","pty","sty"]] = ("u",),
    chunk: int = 512 * 512,
    dtype=None,
    enforce: bool = True,
) -> Tensor:
    # fmt: off
    if dtype is None:
        dtype = gtt.dtype
    out = {}
    for o in outputs:
        out[o] = []

    N = gtt.shape[0]
    for i in range(0, N, chunk):
        A = torch.stack(
            [
                torch.stack([gtt[i:i + chunk], gty[i:i + chunk], gtx[i:i + chunk]], dim=-1),
                torch.stack([gty[i:i + chunk], gyy[i:i + chunk], gyx[i:i + chunk]], dim=-1),
                torch.stack([gtx[i:i + chunk], gyx[i:i + chunk], gxx[i:i + chunk]], dim=-1),
            ],
            dim=-2,
        )
        evals, evecs = torch.linalg.eigh(A.float())
        evals = evals.to(dtype)
        evecs = evecs.to(dtype)
        # evecs: u: [..., 2], v: [..., 1], w: [..., 0]
        # evals: eu: [..., 2], ev: [..., 1], ew: [..., 0]

        if "u" in outputs:
            u = evecs[:, :, 2]
            if enforce:
                u = u * torch.where(u[:, 0:1] < 0, -1.0, 1.0)
            out["u"].append(u.view(-1, 3))
        
        if "v" in outputs:
            v = evecs[:, :, 1]
            if enforce:
                v = v * torch.where(v[:, 1:2] < 0, -1.0, 1.0)
            out["v"].append(v.view(-1, 3))

        if "w" in outputs:
            w = evecs[:, :, 2]
            if enforce:
                w = w * torch.where(w[:, 2:3] < 0, -1.0, 1.0)
            out["w"].append(w.view(-1, 3))

        if "eigs" in outputs:
            out["eigs"].append(evals.view(-1, 3))

    # fmt: on
    for k in out.keys():
        if len(out[k]) != 0:
            out[k] = torch.cat(out[k]).view(*shape, 3)
    return out


def _reorder_vectors(out: dict, taxis: int) -> dict:
    """ensure that the first component of u,v,w is always in the direction specified by taxis"""
    if taxis in (-3, 2):       # (t,y,x)
        perm = [0, 1, 2]
    elif taxis in (-2, 3):     # (y,t,x)
        perm = [1, 0, 2]
    elif taxis in (-1, 4):     # (x,y,t)
        perm = [2, 1, 0]
    else:
        raise ValueError(f"Unsupported taxis={taxis}")

    def _permute(vecs, perm):
        vecs = vecs.unbind(-1)
        return [vecs[i] for i in perm]

    if "u" in out:
        out["u"] = _permute(out["u"], perm)
    if "v" in out:
        out["v"] = _permute(out["v"], perm)
    if "w" in out:
        out["w"] = _permute(out["w"], perm)

    return out



def structure_tensor_orientations3d(
    x: Tensor,
    taxis: int = -1,
    sigma: Union[float, List[float]] = 2.0,
    outputs: Iterable[Literal["u","v","w","eigs","theta","phi","lty","pty","sty"]] = ("u",),
    enforce: bool = True,
    use_fp16: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    outputs should contain:
      - "u"   : List[Tensor], normal vector, 1st (u1), 2nd (u2) and 3rd (u3) components of 1st eigenvector. [u1, u2, u3]
      - "v"   : List[Tensor], 1st (v1), 2nd (v2) and 3rd (v3) components of 2nd eigenvector. [v1, v2, v3]
      - "w"   : List[Tensor], 1st (w1), 2nd (w2) and 3rd (w3) components of 3rd eigenvector. [w1, w2, w3]
      - "eigs": List[Tensor], eigenvalues corresponding to the eigenvectors u, v, w. [eu, ev, ew] and eu >= ev >= ew
      - "theta": Tensor, orientation dip angle
      - "phi": Tensor, orientation azimuthal angle
      - "lty": Tensor, linearity
      - "pty": Tensor, planarity
      - "sty": Tensor, sphericity
    """

    assert x.ndim == 5 and x.shape[1] == 1, "x must be (B,1,T,Y,X)"
    shape = x.shape
    remove_eigs = False
    if "lty" in outputs or 'pty' in outputs or 'sty' in outputs:
        if "eigs" not in outputs:
            outputs += ("eigs", )
            remove_eigs = True

    dtype = x.dtype
    if use_fp16:
        x = x.half()
    
    gt, gy, gx = central_diff3d(x)

    if taxis in (-1, 4):   # t in the last dim
        gx, gt = gt, gx
        if isinstance(sigma, (list, tuple)):
            sigma = list(sigma)
            sigma[-1], sigma[0] = sigma[0], sigma[-1]  # swap t<->x
    elif taxis in (-2, 3):
        gy, gt = gt, gy
        if isinstance(sigma, (list, tuple)):
            sigma = list(sigma)
            sigma[-2], sigma[0] = sigma[0], sigma[-2]  # swap t<->y

    gtt, gyy, gxx, gty, gtx, gyx = _smoothed_products(gt, gy, gx, sigma)

    out = _chunk_eigh(gtt, gyy, gxx, gty, gtx, gyx, shape, outputs, dtype=dtype, enforce=enforce)
    if "theta" in outputs:
        out["theta"] = torch.arccos(out["u"][..., 0])
    if "phi" in outputs:
        out["phi"] = torch.atan2(out["u"][..., 2], out["u"][..., 1])
    if "lty" in outputs or 'pty' in outputs or 'sty' in outputs:
        denom = torch.clamp(out["eigs"][..., 2], min=1e-6)
        if "lty" in outputs:
            out['lty'] = (out["eigs"][..., 2] - out["eigs"][..., 1]) / denom
        if 'pty' in outputs:
            out['pty'] = (out["eigs"][..., 1] - out["eigs"][..., 0]) / denom
        if 'sty' in outputs:
            out['sty'] = out["eigs"][..., 0] / denom

    if remove_eigs:
        out.pop("eigs")
    
    out = _reorder_vectors(out, taxis)

    return out


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    import numpy as np
    import torch

    # def custom_repr(self):
    #     return f'{tuple(self.shape)}'

    # original_repr = torch.Tensor.__repr__
    # torch.Tensor.__repr__ = custom_repr

    root = '/home/jtli/data/seismic_data/f3/subd/'
    sxp = root + 'f3sub_h400x512x512.dat'
    ni, nx, nt = 512, 512, 400
    # p2p = root + '3dp2.dat'
    # epp = root + '3dep.dat'
    u1p = root + '3du1.dat'
    u2p = root + '3du2.dat'
    u3p = root + '3du3.dat'

    def _down(x, scale=2):
        return F.avg_pool3d(x, scale, scale)

    def _up(x, scale=2):
        return F.interpolate(x,
                             scale_factor=scale,
                             mode='trilinear',
                             align_corners=False)

    sx = np.fromfile(sxp, np.float32).reshape(ni, nx, nt)

    x = torch.from_numpy(sx.T).unsqueeze(0).unsqueeze(0).to('cuda')  # (1,1,T,Y,X)
    # x.requires_grad = True
    u = structure_tensor_orientations3d(x, (8, 2, 2), ('u', ), use_fp16=False)["u"]
    print(u.min().item())
    # loss = F.mse_loss(u1*u2, u3)
    # loss.backward()
