# TorchSeis: A Unified 3D Deep Learning Framework for Seismic and Subsurface Modeling

**TorchSeis** is an open research framework that explores the intersection of **deep learning, seismic data analysis, and subsurface intelligence**. It provides reusable models, operators, and experimental tools aimed at advancing AI-driven understanding of the Earth, with applications across:

- Seismic interpretation and structural analysis  (volumetric)
- Faults, stratigraphy, channels, karst, salt bodies, and other geologic features
- Efficient whole-volume inference without patching artifacts for ultra-large 3D datasets  

TorchSeis is intended for both research and industrial deployment, enabling scalable, reproducible, and AI-driven subsurface understanding.

---

## Contents

### 📌 Full-Volume Inference Framework

TorchSeis provides a **retraining-free full-volume inference framework** for 3D dense prediction, enabling seamless processing of datasets up to **1024³** or larger on a single GPU, while preserving spatial continuity and structural coherence.

This work originates from the paper **"Memory-Efficient Full-Volume Inference for Large-Scale 3D Dense Prediction without Performance Degradation"**, which presents an operator-level optimization strategy that enables high-resolution inference on modern GPU hardware without degrading model accuracy.


📄 **Details** can be found at [docs/infer25ce.md](/docs/infer25ce.md)


### 🧭 Seismic Operators & Attributes

- Impedance to Seismic with different wavelets
- Structure Tensor Orientations
  * see [scripts/orientations.ipynb](/scripts/orientations.ipynb) for serval examples.
- Local Slope


## 🔬 Citation

If this repository is useful for your research, please citing the relevant citations below:

```bibtex
@article{li2025infer,
  title={Memory-Efficient Full-Volume Inference for Large-Scale 3D Dense Prediction without Performance Degradation},
  author={Li, Jintao and Wu, Xinming},
  journal={Communications Engineering},
  year={2025}
}
```