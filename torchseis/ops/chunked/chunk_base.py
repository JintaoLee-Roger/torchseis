# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import itertools
import torch
from torch.nn import functional as F


def chunk_forward(
    x,
    ops,
    out_c=None,
    bsize=64,
    pad=0,
    scale=1,
    force_bsize=False,
    padding_mode="constant",
    padding_value=0,
    *args,
    **kwargs,
):
    if isinstance(x, list):
        b, c, d, h, w = x[0].shape
    else:
        b, c, d, h, w = x.shape

    if isinstance(bsize, int):
        bsize = (bsize, bsize, bsize)
    if isinstance(pad, int):
        pad = (pad, pad, pad)
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)

    if out_c is None:
        out_c = c

    tsize = tuple(bs - 2 * p for bs, p in zip(bsize, pad))

    nb_d = (d + tsize[0] - 1) // tsize[0]
    nb_h = (h + tsize[1] - 1) // tsize[1]
    nb_w = (w + tsize[2] - 1) // tsize[2]

    out = torch.zeros(
        (b, out_c, int(d * scale[0]), int(h * scale[1]), int(w * scale[2])),
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

        if force_bsize:
            v_d0 = max(0, v_d1 - tsize[0])
            v_h0 = max(0, v_h1 - tsize[1])
            v_w0 = max(0, v_w1 - tsize[2])

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

        r_d0 = (v_d0 - p_d0 + padlist[0]) * scale[0]
        r_d1 = r_d0 + (v_d1 - v_d0) * scale[0]
        r_h0 = (v_h0 - p_h0 + padlist[1]) * scale[1]
        r_h1 = r_h0 + (v_h1 - v_h0) * scale[1]
        r_w0 = (v_w0 - p_w0 + padlist[2]) * scale[2]
        r_w1 = r_w0 + (v_w1 - v_w0) * scale[2]

        o_d0 = int(v_d0 * scale[0])
        o_h0 = int(v_h0 * scale[1])
        o_w0 = int(v_w0 * scale[2])

        o_d1 = int(v_d1 * scale[0])
        o_h1 = int(v_h1 * scale[1])
        o_w1 = int(v_w1 * scale[2])

        with torch.no_grad():
            patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
            patchi = F.pad(
                patchi,
                padlist,
                mode=padding_mode,
                value=padding_value,
            )
            patcho = ops(patchi, **kwargs)[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

            out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho

    return out
