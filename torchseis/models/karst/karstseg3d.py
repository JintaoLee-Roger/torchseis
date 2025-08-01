"""
KarstSeg3d model is a 3D convolutional U-Net for seismic fault segmentation.

To cite this model:
```bibtex
@article{wu2020deep,
    author = {Wu, Xinming and Yan, Shangsheng and Qi, Jie and Zeng, Hongliu},
    title = {Deep Learning for Characterizing Paleokarst Collapse Features in 3-D Seismic Images},
    journal = {Journal of Geophysical Research: Solid Earth},
    volume = {125},
    number = {9},
    pages = {e2020JB019685},
    doi = {https://doi.org/10.1029/2020JB019685},
    year = {2020}
}
```
"""

from ..fault import FaultSeg3dPlus


class KarstSeg3d(FaultSeg3dPlus):
    """
    Implements KarstSeg3d model from 
    `"Deep Learning for Characterizing Paleokarst Collapse Features in 3-D Seismic Images"
    <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB019685>`_.
    """

    def __init__(self, base: int = 32):
        super().__init__(base=base)
        self.name = "KarstSeg3d"
