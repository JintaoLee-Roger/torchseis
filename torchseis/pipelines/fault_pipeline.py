# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

from torch import nn, Tensor
import torch
import numpy as np
from torch.nn import functional as F
from .base import PipelineBase


class FaultPipeline(PipelineBase):

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.model.eval()
        self.fuse = False
        if self.model.name == "FaultSSL" and self.model.mode == 'precision':
            self.fuse = True

    def preprocess(self, x, *args, **kwargs) -> Tensor:
        if x.ndim != 3 and x.ndim != 5:
            raise ValueError(
                f"Input must be a 3D or 5D tensor, got {x.ndim}D tensor")
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        dtype = x.dtype
        if dtype == torch.float16:
            x = x.float()
        x = (x - x.mean()) / x.std()
        if self.model.name == 'FaultSSL':
            x = torch.clamp(x, -3.2, 3.2)
            x = x / torch.abs(x).max()

        x = x.permute(0, 1, 4, 3, 2)
        x = self._pad_data(x)
        assert x.shape[1] == 1, "Only one channel is supported"

        x = x.to(self.device, dtype=self.dtype)
        return x

    def postprocess(self, x: Tensor, npout=True) -> np.ndarray:
        x = self._unpad_data(x)
        x = x.permute(0, 1, 4, 3, 2)
        if npout:
            x = x.detach().cpu().numpy()[0, 0]
        return x

    @torch.no_grad()
    def __call__(self,
                 seis,
                 rank: int = 0,
                 infer_method: str = 'default',
                 return_all: bool = False,
                 *args,
                 **kwargs) -> Tensor:
        """
        seis: np.ndarray or torch.Tensor, dim (ni, nx, nt) or (nx, ny, nz)
        infer_method: str, default, multi_angle, multi_resolution, multi_angle_resolution
        """
        if infer_method == 'default':
            return self._forward(seis, rank)
        elif infer_method == 'multi_angle':
            return self._multi_angle_forward(seis, rank, return_all)
        elif infer_method == 'multi_resolution':
            return self._multi_resolution_forward(seis, rank, return_all)
        elif infer_method == 'multi_angle_resolution':
            return self._multi_angle_resolution_forward(seis, rank, return_all)
        else:
            raise ValueError(f"Invalid infer method: {infer_method}")

    @torch.no_grad()
    def _forward(self, seis: np.ndarray, rank: int = 0, npout=True) -> Tensor:
        inp = self.preprocess(seis)

        if self.fuse and rank > 0:
            self.model.fuse_model()

        out = self.model(inp, rank)

        if self.fuse and rank > 0:
            self.model.unfuse_model()

        return self.postprocess(out, npout)

    @torch.no_grad()
    def _multi_angle_forward(
        self,
        seis: np.ndarray,
        rank: int = 0,
        return_all: bool = False,
    ) -> Tensor:
        print('--- Multi Angle Infering ---')
        print('1. Normal Infering')
        out1 = self._forward(seis, rank)
        print('2. Infering after rotating 90')
        out2 = self._forward(np.rot90(seis, k=1, axes=(0, 1)).copy(), rank)
        out2 = np.rot90(out2, k=-1, axes=(0, 1))
        print('3. Infering after rotating 180')
        out3 = self._forward(np.rot90(seis, k=2, axes=(0, 1)).copy(), rank)
        out3 = np.rot90(out3, k=-2, axes=(0, 1))
        print('4. Infering after rotating 270')
        out4 = self._forward(np.rot90(seis, k=3, axes=(0, 1)).copy(), rank)
        out4 = np.rot90(out4, k=-3, axes=(0, 1))
        print('5. Merge')
        if return_all:
            return np.maximum.reduce([out1, out2, out3, out4]), {
                '0': out1,
                '90': out2,
                '180': out3,
                '270': out4
            }
        return np.maximum.reduce([out1, out2, out3, out4])

    @torch.no_grad()
    def _multi_resolution_forward(
        self,
        seis: np.ndarray,
        rank: int = 0,
        return_all: bool = False,
    ) -> Tensor:
        print('--- Multi Resolution Infering ---')

        print('1. Normal Infering')
        d, h, w = seis.shape
        out1 = self._forward(seis, rank)

        print('2. Infering after reducing resolution of dimension iline')
        out2 = self._forward(seis[::2, :, :], rank, False)
        out2 = F.interpolate(
            out2,
            (d, h, w),
            mode='trilinear',
        ).detach().cpu().numpy()[0, 0]

        print('3. Infering after reducing resolution of dimension xline')
        out3 = self._forward(seis[:, ::2, :], rank, False)
        out3 = F.interpolate(
            out3,
            (d, h, w),
            mode='trilinear',
        ).detach().cpu().numpy()[0, 0]

        print('4. Infering after reducing resolution of dimension time')
        out4 = self._forward(seis[:, :, ::2], rank, False)
        out4 = F.interpolate(
            out4,
            (d, h, w),
            mode='trilinear',
        ).detach().cpu().numpy()[0, 0]

        print('5. Merge')
        if return_all:
            return np.maximum.reduce([out1, out2, out3, out4]), {
                'normal': out1,
                'reduced_i': out2,
                'reduced_x': out3,
                'reduced_t': out4
            }
        return np.maximum.reduce([out1, out2, out3, out4])

    @torch.no_grad()
    def _multi_angle_resolution_forward(
        self,
        seis: Tensor,
        rank: int = 0,
        return_all: bool = False,
    ) -> Tensor:
        print('--- Multi Angle and Resolution Infering ---')
        out1 = self._multi_angle_forward(seis, rank, return_all)
        out2 = self._multi_resolution_forward(seis, rank, return_all)
        print("Merge multi angle and resolution")
        if return_all:
            return np.maximum.reduce([out1[0], out2[0]]), {
                'multi_angle': out1[1],
                'multi_resolution': out2[1]
            }
        return np.maximum.reduce([out1, out2])

    def _get_padding_of_input(self) -> int:
        if hasattr(self.model, 'need_padding'):
            return self.model.need_padding
        else:
            return 0

    def _pad_data(self, seis: Tensor) -> Tensor:
        tpad = self._get_padding_of_input()
        b, c, d, h, w = seis.shape
        pd, ph, pw = 0, 0, 0
        if d % tpad != 0:
            pd = tpad - d % tpad
        if h % tpad != 0:
            ph = tpad - h % tpad
        if w % tpad != 0:
            pw = tpad - w % tpad
        self.pad_indices = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2,
                            pd // 2, pd - pd // 2)
        # self.input_shape = (b, c, d, h, w)
        seis = F.pad(seis, self.pad_indices)
        return seis

    def _unpad_data(self, seis: Tensor) -> Tensor:
        pw1, pw2, ph1, ph2, pd1, pd2 = self.pad_indices
        pd2 = -pd2 if pd2 > 0 else None
        ph2 = -ph2 if ph2 > 0 else None
        pw2 = -pw2 if pw2 > 0 else None
        return seis[:, :, pd1:pd2, ph1:ph2, pw1:pw2]
