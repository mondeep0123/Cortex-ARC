"""Data loading and processing module."""

from .loader import ARCDataset, load_arc1, load_arc2, load_all
# from .augmentation import augment_task, AugmentationPipeline  # TODO: implement
from .preprocessing import preprocess_grid, normalize_colors

__all__ = [
    "ARCDataset",
    "load_arc1",
    "load_arc2", 
    "load_all",
    # "augment_task",
    # "AugmentationPipeline",
    "preprocess_grid",
    "normalize_colors",
]

