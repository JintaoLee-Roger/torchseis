# Full-Volume Inference without Performance Degradation

### 📝 Paper Title
**Memory-Efficient Full-Volume Inference for Large-Scale 3D Dense Prediction without Performance Degradation** in **Communications Engineering** (2025) by Jintao Li and Xinming Wu

This work introduces a scalable inference framework that enables **whole-volume 3D prediction without accuracy loss**, even on extremely large seismic datasets.  
The approach restructures high-memory operators during inference only (no retraining required), allowing models to process volumes up to `1024³` directly on modern GPUs.

The method is particularly useful for seismic interpretation tasks such as fault detection, RGT estimation, implicit structural modeling, and geological feature segmentation.

---

## 🔑 Key Features

✔ **Retraining-free**: works with existing pretrained models  
✔ **Whole-volume inference**: no ghost boundaries or stitching artifacts  
✔ **Memory-efficient**: reduces decoder/stem memory footprint  
✔ **Faster runtime**: avoids slow CuDNN kernel fallback  
✔ **Operator-level optimization**: convolution, interpolation, and normalization  

---

## 🚀 Basic Usage

Below is an example using `FaultSeg3D` under the TorchSeis framework:

```python
import torch
from torchseis import models as zoo

# 1. Load model
model = zoo.FaultSeg3d()

# 2. Load pretrained weights
state = torch.load('faultseg3d-2020-70.pth', weights_only=True)
model.load_state_dict(state)

# 3. Convert to GPU
model = model.half().eval().cuda()

# 4. Prepare input volume 
data = torch.from_numpy(f3d[np.newaxis, np.newaxis].copy()).half().cuda()

# 5. Full-volume inference (no tiling)
with torch.no_grad():
    pred = model(data, rank=3).cpu().numpy()
```

> `rank=3` means that using strategy 4 in the paper.

---

## 🧠 Model Zoo Compatibility

The full-volume inference method is compatible with the following TorchSeis models:

| Model            | Task                                           | Source           |
| ---------------- | ---------------------------------------------- | ---------------- |
| `FaultSeg3d`     | Fault segmentation                             | [Wu, et, al., 2019, Geophysics](https://library.seg.org/doi/abs/10.1190/geo2018-0646.1)   |
| `FaultSeg3dPlus` | Fault segmentation                             | [Li, et, al., 2024, Geophysics](https://library.seg.org/doi/abs/10.1190/geo2022-0778.1)   |
| `FaultSSL`       | Fault segmentation                             | [Dou, et, al., 2024, Geophysics](https://library.seg.org/doi/abs/10.1190/geo2023-0550.1)   |
| `Bi21RGT3d`      | Relative geological time (RGT) Estimation      | [Bi, et, al., 2021, JGR-SE](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB021882)   |
| `DeepISMNet`     | Implicit structural modeling                   | [Bi, et, al., 2022, GMD](https://gmd.copernicus.org/articles/15/6841/2022/)  |
| `ChannelSeg3d`   | Channel segmentation                           | [Gao, et, al., 2021, Geophysics](https://library.seg.org/doi/abs/10.1190/geo2020-0572.1)   |
| `Wang25Channel`  | Channel segmentation                           | [Wang, et, al., 2025, ESSD](https://essd.copernicus.org/articles/17/3447/2025)   |
| `KarstSeg3d`     | Paleokarst detection                           | [Wu, et, al., 2020, JGR-SE](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB019685)  |
| `GEM`            | Geological Everything Model                    | [Dou, et, al. 2025](https://arxiv.org/abs/2507.00419) |
| `SegFormer3D`    | 3D medical segmentation (Transformer)          | [Perera, et, al., 2025, CVPR](https://arxiv.org/abs/2404.10156)   |

> Some Transformer-based models may have partial optimization benefit due to global attention memory limits.

---

## 📊 Results Summary


---

## 📚 Citation

If this work or the inference strategy is used in your research, please cite:

```bibtex
@article{li2025infer,
  title={Memory-Efficient Full-Volume Inference for Large-Scale 3D Dense Prediction without Performance Degradation},
  author={Li, Jintao and Wu, Xinming},
  journal={Communications Engineering},
  year={2025}
}
```

---

For questions or contributions, please submit an issue or pull request via the main repository.
