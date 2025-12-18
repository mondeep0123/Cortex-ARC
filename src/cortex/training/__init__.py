"""Training components."""

from .color_tasks import ColorTaskGenerator, ColorDataLoader
from .trainer import Trainer

__all__ = [
    'ColorTaskGenerator',
    'ColorDataLoader',
    'Trainer',
]
