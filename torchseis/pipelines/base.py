# Copyright (c) 2025 Jintao Li.
# University of Science and Technology of China (USTC).
# All rights reserved.

import torch


class PipelineBase:

    def to(self, device, dtype=None):
        self.model.to(device)
        self.device = device
        if dtype is not None:
            self.dtype = dtype
            self.model.to(dtype=dtype)
        return self

    def half(self):
        self.model.half()
        self.dtype = torch.float16
        return self

    def float(self):
        self.model.float()
        self.dtype = torch.float32
        return self

    def cuda(self):
        self.model.cuda()
        self.device = torch.device("cuda")
        return self

    def cpu(self):
        self.model.cpu()
        self.device = torch.device("cpu")
        return self
