"""ARC-AGI Solver with Curriculum Learning."""

__version__ = "0.1.0"
__author__ = "ARC-AGI Research"

from .core import (
    Task,
    SkillOutput,
    SkillModule,
    SkillLibrary,
    ObjectCognitionSkill,
)

from .utils import (
    Grid,
    create_grid,
    find_objects,
    apply_transform,
)

__all__ = [
    'Task',
    'SkillOutput',
    'SkillModule',
    'SkillLibrary',
    'ObjectCognitionSkill',
    'Grid',
    'create_grid',
    'find_objects',
    'apply_transform',
]
