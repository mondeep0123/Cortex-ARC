"""
Visual Cortex Module - Phase 3 Implementation

Implements the visual processing pipeline inspired by the human visual cortex:

    V1 (Primary Visual Cortex):
        - Color Encoder: Understand what colors mean (0-9)
        - Position Encoder: Know where each cell is in the grid
        
    V2 (Secondary Visual Cortex):
        - Edge Detector: Find boundaries between colors
        
    V4 (Higher Visual Areas):
        - Region Detector: Find connected components
        - Shape Recognizer: Identify basic shapes
        
    IT (Inferotemporal Cortex):
        - Pattern Detector: Find repeating patterns
        - Integrator: Combine all features

Phase 3 Implementation:
    - perception.py: VisualCortex, Object, VisualOutput
    - reasoning.py: ObjectMatcher, ComparisonModule, TransformationRule
    - solver.py: Phase3Solver
"""

from .color_encoder import ColorEncoder, ARC_COLORS, COLOR_PROPERTIES

# Phase 3 Components
from .perception import (
    VisualCortex,
    VisualOutput, 
    Object,
    compute_displacement,
    objects_same_shape,
)

from .reasoning import (
    ObjectMatcher,
    ObjectCorrespondence,
    ObjectMatch,
    ComparisonModule,
    TransformationSignature,
    TransformationRule,
    TransformType,
    MatcherConfig,
    extract_rules,
)

from .solver import (
    Phase3Solver,
    Task,
    TrainExample,
    create_test_task,
)

__all__ = [
    # Original
    "ColorEncoder",
    "ARC_COLORS",
    "COLOR_PROPERTIES",
    # Phase 3 Perception
    "VisualCortex",
    "VisualOutput",
    "Object",
    "compute_displacement",
    "objects_same_shape",
    # Phase 3 Reasoning
    "ObjectMatcher",
    "ObjectCorrespondence",
    "ObjectMatch",
    "ComparisonModule",
    "TransformationSignature",
    "TransformationRule",
    "TransformType",
    "MatcherConfig",
    "extract_rules",
    # Phase 3 Solver
    "Phase3Solver",
    "Task",
    "TrainExample",
    "create_test_task",
]

