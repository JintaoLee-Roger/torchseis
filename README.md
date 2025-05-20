
# TorchSeis

PyTorch-based seismic toolkit.

## Installation

```bash
git clone https://github.com/yourusername/torchseis.git
cd torchseis
pip install -e .
```

## Quick Start

```python
import numpy as np
import torch
from torchseis import FaultSeg3d, FaultPipeline

model = FaultSeg3d()
params = torch.load('faultseg3d-2020-70.pt', weights_only=True)
model.load_state_dict(params)
pipeline = FaultPipeline(model).half().cuda()

inp = np.random.randn((1024, 1024, 1024), dtype=np.float32)
# cost 26.86 GB, 7.26s on a H20 GPU
out = pipeline(inp, rank=3)
```

## Pretrained Weights

The pretrained weights can be download from [https://rec.ustc.edu.cn/share/5761aa30-25e7-11f0-ac2f-0bb7b749fade](https://rec.ustc.edu.cn/share/5761aa30-25e7-11f0-ac2f-0bb7b749fade) (password: 6y6a)



## Model Cards

| Model                    | rank | shape              | GPU Memory | Inference Time (ms) |
|--------------------------|------|--------------------|------------|---------------------|
| FaultSeg3d               |   3  | (1024, 1024, 1024) | 26.86 GB   | 7.26s               |
| FaultSeg3dPlus           |   3  | (1024, 1024, 1024) | 56.75 GB   | 14.56s              |
| FaultSSL-precision       |   3  | (1024, 1024, 1024) | 57.73 GB   | 53.43s              |
| FaultSSL-precision(fuse) |   3  | (1024, 1024, 1024) | 57.73 GB   | 34.42s              |
| FaultSSL-iou             |   3  | (1024, 1024, 1024) | 58.19 GB   | 162.1s              |