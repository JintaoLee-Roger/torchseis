# Copyright (c) 2025 Jintao Li.
# Zhejiang University (ZJU).
# University of Science and Technology of China (USTC).
# All rights reserved.

import itertools
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ...ops.chunked.conv import set_pad_as_zero
from ...ops.chunked.norms import AdaptiveInstanceNorm3d
from ...ops.fuse import ConvInstanceNorm3d


def get_index(i, j, k, bsize, shape, pad, scale=1):
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


class FaultSSLChunk:

    def _input_layer1_chunk(self, x, padlist, t=-1, indices=None):
        x = self.input_layer1[0](x)
        if t == 0:
            self.input_layer1[1].stats.add_batch(x[indices])
            return x
        elif self.mode == 'iou' and not self.input_layer1[
                1].stats.has_statistics():
            self.input_layer1[1].stats.compute()
            if self.fused:
                mean = self.input_layer1[1].stats.mean
                var = self.input_layer1[1].stats.var
                self.input_layer1[0] = ConvInstanceNorm3d(
                    self.input_layer1[0], self.input_layer1[1], mean, var)
                self.input_layer1[1].fused = True

        x = self.input_layer1[1](x)
        x = self.input_layer1[2](x)
        x = set_pad_as_zero(x, padlist)

        x = self.input_layer1[3].chunk_forward(x, padlist, t - 1, indices)
        if self.mode == 'iou' and t >= 0 and t < 4:
            return

        x = self.input_layer1[4].chunk_forward(x, padlist, t - 4, indices)
        return x

    def _decoder2_chunk(self, x, padlist, t=-1, indices1=None, indices2=None):
        padlist2 = [p * 2 for p in padlist]
        x = self.Decoder2[0].chunk_forward(x, padlist, t, indices1)
        if t < 2:
            return

        x = self.Decoder2[1](x)

        if t == 2:
            self.Decoder2[2].stats.add_batch(x[indices2])
            return x
        elif self.mode == 'iou' and not self.Decoder2[2].stats.has_statistics(
        ):
            self.Decoder2[2].stats.compute()

        x = self.Decoder2[2](x)
        x = self.Decoder2[3](x)
        x = set_pad_as_zero(x, padlist2)

        x = self.Decoder2[4].chunk_forward(x, padlist2, t - 3, indices2)
        return x

    def _decoder3_chunk(self, x, padlist, t=-1, indices=None):
        x = self.Decoder3[0].chunk_forward(x, padlist, t, indices)
        if self.mode == 'iou' and t < 2:
            return
        x = self.Decoder3[1](x)
        return x

    def _input_layer12(self, x, bsize=64, down=True):
        b, c, d, h, w = x.shape
        oc = self.base

        if down:
            outc = oc * 2
            pad = 4
            oshape = (b, outc, d // 2, h // 2, w // 2)
            scale = 0.5
        else:
            outc = oc
            pad = 4
            oshape = (b, outc, d, h, w)
            scale = 1

        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        # oshape = (b, 6, d, h, w)
        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)
        if not isinstance(self.input_layer1[1], AdaptiveInstanceNorm3d):
            self.input_layer1[1] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.input_layer1[1])

        self.input_layer1[3].replace_norm()
        self.input_layer1[4].replace_norm()
        if down:
            if not isinstance(self.input_layer2[1], AdaptiveInstanceNorm3d):
                self.input_layer2[
                    1] = AdaptiveInstanceNorm3d.from_instance_norm(
                        self.input_layer2[1])

        nt = 9
        for t in range(nt):

            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable

                if down:
                    indices1 = (slice(None), slice(None),
                                slice(r_d0 * 2,
                                      r_d1 * 2), slice(r_h0 * 2, r_h1 * 2),
                                slice(r_w0 * 2, r_w1 * 2))
                    indices2 = (slice(None), slice(None), slice(r_d0, r_d1),
                                slice(r_h0, r_h1), slice(r_w0, r_w1))
                else:
                    indices1 = (slice(None), slice(None), slice(r_d0, r_d1),
                                slice(r_h0, r_h1), slice(r_w0, r_w1))
                    indices2 = None

                with torch.no_grad():
                    patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    patchi = F.pad(patchi, padlist, mode='constant', value=0)
                    patcho = self._input_layer1_chunk(patchi, padlist, t,
                                                      indices1)
                    if down and t >= nt - 2:
                        patcho = self.input_layer2[0](patcho)
                        if t == nt - 2:
                            self.input_layer2[1].stats.add_batch(
                                patcho[indices2])
                        elif not self.input_layer2[1].stats.has_statistics():
                            self.input_layer2[1].stats.compute()
                            if self.fused:
                                mean = self.input_layer2[1].stats.mean
                                var = self.input_layer2[1].stats.var
                                self.input_layer2[0] = ConvInstanceNorm3d(
                                    self.input_layer2[0], self.input_layer2[1],
                                    mean, var)
                                self.input_layer2[1].fused = True

                        if t == nt - 1:
                            patcho = self.input_layer2[1:3](patcho)

                    if t == nt - 1:
                        out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1] # yapf: disable

        if down:
            self.input_layer2[1].stats.reset()
            self.input_layer2[1].fused = False
        return out

    def _input_layer1(self,
                      x: Tensor,
                      bsize: int = 64,
                      down: bool = True) -> Tensor:
        b, c, d, h, w = x.shape
        oc = self.base

        if down:
            outc = oc * 2
            pad = 4
            oshape = (b, outc, d // 2, h // 2, w // 2)
            scale = 0.5
        else:
            outc = oc
            pad = 4
            oshape = (b, outc, d, h, w)
            scale = 1

        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize

        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, scale) # yapf: disable

            with torch.no_grad():
                patchi = x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patcho = self._input_layer1_chunk(patchi, padlist)

                if down:
                    patcho = self.input_layer2[0:3](patcho)
                out[:, :, o_d0:o_d1, o_h0:o_h1,
                    o_w0:o_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1]

        return out

    def _decoder232(
        self,
        x: Tensor,
        skip2: Tensor,
        skip1: Tensor = None,
        inp: Tensor = None,
    ) -> Tensor:
        assert inp is not None or skip1 is not None
        recomp = True
        if skip1 is not None:
            recomp = False
        b, c, d, h, w = x.shape
        oshape = (b, 1, d * 2, h * 2, w * 2)
        bsize = 32
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize
        pad = 2
        out = torch.zeros_like(x)
        # out = torch.zeros((b, 24, d*2, h*2, w*2), dtype=x.dtype, device=x.device)

        self.Decoder2[0].replace_norm()
        self.Decoder2[4].replace_norm()

        if not isinstance(self.Decoder2[2], AdaptiveInstanceNorm3d):
            self.Decoder2[2] = AdaptiveInstanceNorm3d.from_instance_norm(
                self.Decoder2[2])

        nt = 7
        for t in range(nt):
            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
                v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, 1) # yapf: disable

                indices1 = (slice(None), slice(None), slice(r_d0, r_d1),
                            slice(r_h0, r_h1), slice(r_w0, r_w1))

                indices2 = (slice(None), slice(None),
                            slice(r_d0 * 2,
                                  r_d1 * 2), slice(r_h0 * 2, r_h1 * 2),
                            slice(r_w0 * 2, r_w1 * 2))

                with torch.no_grad():
                    patchi = torch.cat([
                        x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1],
                        skip2[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                    ], 1)
                    patchi = F.pad(patchi, padlist, mode='constant', value=0)
                    if t == nt - 1:
                        patcho = self.Decoder2[0].chunk_forward(
                            patchi, padlist, t, indices1)
                        out[:, :, o_d0:o_d1, o_h0:o_h1, o_w0:o_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1] # yapf: disable
                    else:
                        patcho = self._decoder2_chunk(patchi, padlist, t,
                                                      indices1, indices2)

        # return out
        x = out
        self.Decoder3[0].replace_norm()
        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        # decoder2, input_layer1, decoder3
        # v3_d0,p3_d0 -> decoder3 (pad3)[v3_d0]
        # p3_d0 -> v2_d0,p2_d0 -> input_layer1 (pad2)[p3_d0]
        # v3_d0//2 -> v1_d0,p1_d0 -> decoder2 (pad1)[p3_d0]
        pad1, pad2, pad3 = 2, 4, 4

        bsize = 64
        nb_d = (d * 2 + bsize - 1) // bsize
        nb_h = (h * 2 + bsize - 1) // bsize
        nb_w = (w * 2 + bsize - 1) // bsize
        nt = 3
        for t in range(nt):
            for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
                v3_d0, v3_h0, v3_w0, v3_d1, v3_h1, v3_w1, p3_d0, p3_h0, p3_w0, p3_d1, p3_h1, p3_w1, r3_d0, r3_h0, r3_w0, r3_d1, r3_h1, r3_w1, o3_d0, o3_h0, o3_w0, o3_d1, o3_h1, o3_w1, padlist3 = get_index(i, j, k, bsize, (d * 2, h * 2, w * 2), pad3, 1) # yapf: disable

                v2_d0 = p3_d0
                v2_h0 = p3_h0
                v2_w0 = p3_w0

                v2_d1 = p3_d1
                v2_h1 = p3_h1
                v2_w1 = p3_w1

                p2_d0 = max(0, v2_d0 - pad2)
                p2_h0 = max(0, v2_h0 - pad2)
                p2_w0 = max(0, v2_w0 - pad2)

                p2_d1 = min(d * 2, v2_d1 + pad2)
                p2_h1 = min(h * 2, v2_h1 + pad2)
                p2_w1 = min(w * 2, v2_w1 + pad2)

                padlist2 = [0, 0, 0, 0, 0, 0]

                if p2_d1 == v2_d1:
                    padlist2[5] = pad2
                if p2_h1 == v2_h1:
                    padlist2[3] = pad2
                if p2_w1 == v2_w1:
                    padlist2[1] = pad2
                if p2_d0 == v2_d0:
                    padlist2[4] = pad2
                if p2_h0 == v2_h0:
                    padlist2[2] = pad2
                if p2_w0 == v2_w0:
                    padlist2[0] = pad2

                r2_d0 = v2_d0 - p2_d0 + padlist2[4]
                r2_d1 = r2_d0 + (v2_d1 - v2_d0)
                r2_h0 = (v2_h0 - p2_h0 + padlist2[2])
                r2_h1 = r2_h0 + (v2_h1 - v2_h0)
                r2_w0 = (v2_w0 - p2_w0 + padlist2[0])
                r2_w1 = r2_w0 + (v2_w1 - v2_w0)

                v1_d0 = p3_d0 // 2
                v1_h0 = p3_h0 // 2
                v1_w0 = p3_w0 // 2

                v1_d1 = p3_d1 // 2
                v1_h1 = p3_h1 // 2
                v1_w1 = p3_w1 // 2

                p1_d0 = max(0, v1_d0 - pad1)
                p1_h0 = max(0, v1_h0 - pad1)
                p1_w0 = max(0, v1_w0 - pad1)

                p1_d1 = min(d, v1_d1 + pad1)
                p1_h1 = min(h, v1_h1 + pad1)
                p1_w1 = min(w, v1_w1 + pad1)

                padlist1 = [0, 0, 0, 0, 0, 0]

                if p1_d1 == v1_d1:
                    padlist1[5] = pad1
                if p1_h1 == v1_h1:
                    padlist1[3] = pad1
                if p1_w1 == v1_w1:
                    padlist1[1] = pad1
                if p1_d0 == v1_d0:
                    padlist1[4] = pad1
                if p1_h0 == v1_h0:
                    padlist1[2] = pad1
                if p1_w0 == v1_w0:
                    padlist1[0] = pad1

                r1_d0 = (v1_d0 - p1_d0 + padlist1[4]) * 2
                r1_d1 = r1_d0 + ((v1_d1 - v1_d0)) * 2
                r1_h0 = (v1_h0 - p1_h0 + padlist1[2]) * 2
                r1_h1 = r1_h0 + ((v1_h1 - v1_h0)) * 2
                r1_w0 = (v1_w0 - p1_w0 + padlist1[0]) * 2
                r1_w1 = r1_w0 + ((v1_w1 - v1_w0)) * 2
                padlist12 = [p * 2 for p in padlist1]

                with torch.no_grad():
                    patchi = x[:, :, p1_d0:p1_d1, p1_h0:p1_h1, p1_w0:p1_w1]
                    patchi = F.pad(patchi, padlist1, mode='constant', value=0)
                    patchi = self.Decoder2[1:4](patchi)
                    patchi = set_pad_as_zero(patchi, padlist12)
                    patchi = self.Decoder2[4].chunk_forward(patchi, padlist12)
                    patchi = patchi[:, :, r1_d0:r1_d1, r1_h0:r1_h1, r1_w0:r1_w1]

                    if recomp:
                        patchj = inp[:, :, p2_d0:p2_d1, p2_h0:p2_h1, p2_w0:p2_w1]
                        patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                        patchj = self._input_layer1_chunk(patchj, padlist2)
                        patchj = patchj[:, :, r2_d0:r2_d1, r2_h0:r2_h1, r2_w0:r2_w1]
                    else:
                        patchj = skip1[:, :, p2_d0:p2_d1, p2_h0:p2_h1, p2_w0:p2_w1]
                        patchj = F.pad(patchj, padlist2, mode='constant', value=0) 
                        patchj = patchj[:, :, r2_d0:r2_d1, r2_h0:r2_h1, r2_w0:r2_w1]

                    patcho = torch.cat([patchi, patchj], 1)
                    patcho = F.pad(patcho, padlist3, mode='constant', value=0)
                    indices = (slice(None), slice(None), slice(r3_d0, r3_d1), slice(r3_h0, r3_h1), slice(r3_w0, r3_w1))

                    patcho = self._decoder3_chunk(patcho, padlist3, t, indices)
                    if t == nt - 1:
                        out[:, :, o3_d0:o3_d1, o3_h0:o3_h1, o3_w0:o3_w1] = patcho[:, :, r3_d0:r3_d1, r3_h0:r3_h1, r3_w0:r3_w1] # yapf: disable

        self.input_layer1[1].stats.reset()
        self.input_layer1[1].fused = False
        self.input_layer1[3].reset()
        self.input_layer1[4].reset()
        self.Decoder2[0].reset()
        self.Decoder2[2].stats.reset()
        self.Decoder2[4].reset()
        self.Decoder3[0].reset()
        return out

    def _decoder23(
        self,
        x: Tensor,
        skip2: Tensor,
        skip1=None,
        inp: Tensor = None,
    ) -> Tensor:
        assert inp is not None or skip1 is not None
        recomp = True
        if skip1 is not None:
            recomp = False
        b, c, d, h, w = x.shape
        oshape = (b, 1, d * 2, h * 2, w * 2)
        out = torch.zeros_like(x)
        bsize = 64
        nb_d = (d + bsize - 1) // bsize
        nb_h = (h + bsize - 1) // bsize
        nb_w = (w + bsize - 1) // bsize
        pad = 2

        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
            v_d0, v_h0, v_w0, v_d1, v_h1, v_w1, p_d0, p_h0, p_w0, p_d1, p_h1, p_w1, r_d0, r_h0, r_w0, r_d1, r_h1, r_w1, o_d0, o_h0, o_w0, o_d1, o_h1, o_w1, padlist = get_index(i, j, k, bsize, (d, h, w), pad, 1) # yapf: disable

            with torch.no_grad():
                patchi = torch.cat([
                    x[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1],
                    skip2[:, :, p_d0:p_d1, p_h0:p_h1, p_w0:p_w1]
                ], 1)
                patchi = F.pad(patchi, padlist, mode='constant', value=0)
                patcho = self.Decoder2[0].chunk_forward(patchi, padlist)
                out[:, :, v_d0:v_d1, v_h0:v_h1, v_w0:v_w1] = patcho[:, :, r_d0:r_d1, r_h0:r_h1, r_w0:r_w1] # yapf: disable

        x = out
        out = torch.zeros(oshape, device=x.device, dtype=x.dtype)

        # decoder2, input_layer1, decoder3
        # v3_d0,p3_d0 -> decoder3 (pad3)[v3_d0]
        # p3_d0 -> v2_d0,p2_d0 -> input_layer1 (pad2)[p3_d0]
        # v3_d0//2 -> v1_d0,p1_d0 -> decoder2 (pad1)[p3_d0]
        pad1, pad2, pad3 = 2, 4, 4

        bsize = 128
        nb_d = (d * 2 + bsize - 1) // bsize
        nb_h = (h * 2 + bsize - 1) // bsize
        nb_w = (w * 2 + bsize - 1) // bsize
        for i, j, k in itertools.product(range(nb_d), range(nb_h), range(nb_w)): # yapf: disable
            v3_d0, v3_h0, v3_w0, v3_d1, v3_h1, v3_w1, p3_d0, p3_h0, p3_w0, p3_d1, p3_h1, p3_w1, r3_d0, r3_h0, r3_w0, r3_d1, r3_h1, r3_w1, o3_d0, o3_h0, o3_w0, o3_d1, o3_h1, o3_w1, padlist3 = get_index(i, j, k, bsize, (d * 2, h * 2, w * 2), pad3, 1) # yapf: disable

            v2_d0 = p3_d0
            v2_h0 = p3_h0
            v2_w0 = p3_w0

            v2_d1 = p3_d1
            v2_h1 = p3_h1
            v2_w1 = p3_w1

            p2_d0 = max(0, v2_d0 - pad2)
            p2_h0 = max(0, v2_h0 - pad2)
            p2_w0 = max(0, v2_w0 - pad2)

            p2_d1 = min(d * 2, v2_d1 + pad2)
            p2_h1 = min(h * 2, v2_h1 + pad2)
            p2_w1 = min(w * 2, v2_w1 + pad2)

            padlist2 = [0, 0, 0, 0, 0, 0]

            if p2_d1 == v2_d1:
                padlist2[5] = pad2
            if p2_h1 == v2_h1:
                padlist2[3] = pad2
            if p2_w1 == v2_w1:
                padlist2[1] = pad2
            if p2_d0 == v2_d0:
                padlist2[4] = pad2
            if p2_h0 == v2_h0:
                padlist2[2] = pad2
            if p2_w0 == v2_w0:
                padlist2[0] = pad2

            r2_d0 = v2_d0 - p2_d0 + padlist2[4]
            r2_d1 = r2_d0 + (v2_d1 - v2_d0)
            r2_h0 = (v2_h0 - p2_h0 + padlist2[2])
            r2_h1 = r2_h0 + (v2_h1 - v2_h0)
            r2_w0 = (v2_w0 - p2_w0 + padlist2[0])
            r2_w1 = r2_w0 + (v2_w1 - v2_w0)

            v1_d0 = p3_d0 // 2
            v1_h0 = p3_h0 // 2
            v1_w0 = p3_w0 // 2

            v1_d1 = p3_d1 // 2
            v1_h1 = p3_h1 // 2
            v1_w1 = p3_w1 // 2

            p1_d0 = max(0, v1_d0 - pad1)
            p1_h0 = max(0, v1_h0 - pad1)
            p1_w0 = max(0, v1_w0 - pad1)

            p1_d1 = min(d, v1_d1 + pad1)
            p1_h1 = min(h, v1_h1 + pad1)
            p1_w1 = min(w, v1_w1 + pad1)

            padlist1 = [0, 0, 0, 0, 0, 0]

            if p1_d1 == v1_d1:
                padlist1[5] = pad1
            if p1_h1 == v1_h1:
                padlist1[3] = pad1
            if p1_w1 == v1_w1:
                padlist1[1] = pad1
            if p1_d0 == v1_d0:
                padlist1[4] = pad1
            if p1_h0 == v1_h0:
                padlist1[2] = pad1
            if p1_w0 == v1_w0:
                padlist1[0] = pad1

            r1_d0 = (v1_d0 - p1_d0 + padlist1[4]) * 2
            r1_d1 = r1_d0 + ((v1_d1 - v1_d0)) * 2
            r1_h0 = (v1_h0 - p1_h0 + padlist1[2]) * 2
            r1_h1 = r1_h0 + ((v1_h1 - v1_h0)) * 2
            r1_w0 = (v1_w0 - p1_w0 + padlist1[0]) * 2
            r1_w1 = r1_w0 + ((v1_w1 - v1_w0)) * 2
            padlist12 = [p * 2 for p in padlist1]

            with torch.no_grad():
                patchi = x[:, :, p1_d0:p1_d1, p1_h0:p1_h1, p1_w0:p1_w1]
                patchi = F.pad(patchi, padlist1, mode='constant', value=0)
                patchi = self.Decoder2[1:4](patchi)
                patchi = set_pad_as_zero(patchi, padlist12)
                patchi = self.Decoder2[4].chunk_forward(patchi, padlist12)
                patchi = patchi[:, :, r1_d0:r1_d1, r1_h0:r1_h1, r1_w0:r1_w1]

                if recomp:
                    patchj = inp[:, :, p2_d0:p2_d1, p2_h0:p2_h1, p2_w0:p2_w1]
                    patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                    patchj = self._input_layer1_chunk(patchj, padlist2)
                    patchj = patchj[:, :, r2_d0:r2_d1, r2_h0:r2_h1, r2_w0:r2_w1]
                else:
                    patchj = skip1[:, :, p2_d0:p2_d1, p2_h0:p2_h1, p2_w0:p2_w1]
                    patchj = F.pad(patchj, padlist2, mode='constant', value=0)
                    patchj = patchj[:, :, r2_d0:r2_d1, r2_h0:r2_h1, r2_w0:r2_w1]

                # patcho = torch.cat([patchi, patchj], 1)
                # out[:, :, p3_d0:p3_d1, p3_h0:p3_h1, p3_w0:p3_w1] = patcho

                patcho = torch.cat([patchi, patchj], 1)
                patcho = F.pad(patcho, padlist3, mode='constant', value=0)
                patcho = self._decoder3_chunk(patcho, padlist3)
                out[:, :, o3_d0:o3_d1, o3_h0:o3_h1, o3_w0:o3_w1] = patcho[:, :, r3_d0:r3_d1, r3_h0:r3_h1, r3_w0:r3_w1] # yapf: disable

        return out
