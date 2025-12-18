"""
Cortex: Unified Reasoning Model

A single neural network that learns all cognitive abilities:
- Color understanding
- Spatial reasoning
- Pattern recognition
- Object detection
- Relational reasoning

All abilities emerge from shared weights through curriculum training.
"""

from .model.cortex import CortexModel
from .model.encoder import GridEncoder
from .model.decoder import GridDecoder

__all__ = [
    'CortexModel',
    'GridEncoder', 
    'GridDecoder',
]
