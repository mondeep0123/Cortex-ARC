"""
Object Cognition Skill Module

This module implements object-level reasoning:
- Object detection and segmentation (connected components)
- Object tracking and identity
- Object boundaries and cohesion
- Object permanence

This is a foundational skill used by many higher-level reasoning tasks.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from .base import SkillModule, SkillOutput, Task
    from ..utils import grid_utils
except ImportError:
    from src.core.base import SkillModule, SkillOutput, Task
    from src.utils import grid_utils

# Type aliases
Grid = np.ndarray


@dataclass
class ObjectAnalysis:
    """Analysis of objects in a grid."""
    objects: List[grid_utils.GridObject]
    background_color: int
    num_objects: int
    colors_present: set
    largest_object: Optional[grid_utils.GridObject] = None
    
    def __post_init__(self):
        if self.objects:
            self.largest_object = max(self.objects, key=lambda obj: obj.size)


class ObjectCognitionSkill(SkillModule):
    """
    Skill for understanding and manipulating objects in grids.
    
    Capabilities:
    - Detect connected components
    - Extract individual objects
    - Filter objects by properties (color, size, shape)
    - Compose objects into new arrangements
    """
    
    def __init__(self):
        super().__init__("object_cognition")
        self.connectivity = 4  # 4 or 8-connectivity for object detection
    
    def analyze_objects(self, grid: Grid) -> ObjectAnalysis:
        """Analyze all objects in a grid."""
        background = grid_utils.get_background_color(grid)
        objects = grid_utils.find_objects(grid, background=background, connectivity=self.connectivity)
        colors = set(obj.color for obj in objects)
        
        return ObjectAnalysis(
            objects=objects,
            background_color=background,
            num_objects=len(objects),
            colors_present=colors
        )
    
    def apply(self, grid: Grid, context: Optional[Dict[str, Any]] = None) -> SkillOutput:
        """
        Apply object cognition to a grid.
        
        Default behavior: detect and segment all objects.
        Context can specify specific operations.
        """
        if context is None:
            context = {}
        
        operation = context.get("operation", "detect")
        
        if operation == "detect":
            return self._detect_objects(grid)
        elif operation == "extract_largest":
            return self._extract_largest(grid)
        elif operation == "extract_by_color":
            target_color = context.get("color", 1)
            return self._extract_by_color(grid, target_color)
        elif operation == "count":
            return self._count_objects(grid)
        elif operation == "isolate":
            # Keep each object in separate grid
            return self._isolate_objects(grid)
        else:
            return self._detect_objects(grid)
    
    def _detect_objects(self, grid: Grid) -> SkillOutput:
        """Detect all objects and return visualization."""
        analysis = self.analyze_objects(grid)
        
        # Create visualization with each object highlighted
        result = grid.copy()
        
        reasoning = (f"Detected {analysis.num_objects} objects with "
                    f"{len(analysis.colors_present)} different colors")
        
        return SkillOutput(
            result=result,
            confidence=0.95,
            reasoning=reasoning
        )
    
    def _extract_largest(self, grid: Grid) -> SkillOutput:
        """Extract the largest object."""
        analysis = self.analyze_objects(grid)
        
        if not analysis.objects:
            return SkillOutput(
                result=grid_utils.create_grid(1, 1, 0),
                confidence=0.0,
                reasoning="No objects found"
            )
        
        largest = analysis.largest_object
        
        # Extract object to tight bounding box
        r_min, r_max, c_min, c_max = largest.bbox
        extracted = grid[r_min:r_max+1, c_min:c_max+1].copy()
        
        # Keep only the largest object, set others to background
        background = analysis.background_color
        for r in range(extracted.shape[0]):
            for c in range(extracted.shape[1]):
                abs_r, abs_c = r_min + r, c_min + c
                if (abs_r, abs_c) not in largest.positions:
                    extracted[r, c] = background
        
        reasoning = (f"Extracted largest object (size={largest.size}, "
                    f"color={largest.color})")
        
        return SkillOutput(
            result=extracted,
            confidence=0.9,
            reasoning=reasoning
        )
    
    def _extract_by_color(self, grid: Grid, target_color: int) -> SkillOutput:
        """Extract all objects of a specific color."""
        analysis = self.analyze_objects(grid)
        
        # Find objects with target color
        matching = [obj for obj in analysis.objects if obj.color == target_color]
        
        if not matching:
            return SkillOutput(
                result=grid_utils.create_grid(1, 1, 0),
                confidence=0.0,
                reasoning=f"No objects with color {target_color}"
            )
        
        # Create result with only matching objects
        result = grid_utils.create_grid(grid.shape[0], grid.shape[1], analysis.background_color)
        
        for obj in matching:
            for r, c in obj.positions:
                result[r, c] = grid[r, c]
        
        reasoning = f"Extracted {len(matching)} objects with color {target_color}"
        
        return SkillOutput(
            result=result,
            confidence=0.85,
            reasoning=reasoning
        )
    
    def _count_objects(self, grid: Grid) -> SkillOutput:
        """Count objects and represent as grid."""
        analysis = self.analyze_objects(grid)
        
        # Create a small grid representing the count
        count = analysis.num_objects
        
        # Represent count as a 1D grid (simple representation)
        # In real implementation, might encode count more sophisticatedly
        result = grid_utils.create_grid(1, count, fill_color=1)
        
        reasoning = f"Counted {count} objects"
        
        return SkillOutput(
            result=result,
            confidence=0.95,
            reasoning=reasoning
        )
    
    def _isolate_objects(self, grid: Grid) -> SkillOutput:
        """Isolate each object in its own minimal bounding box."""
        analysis = self.analyze_objects(grid)
        
        if not analysis.objects:
            return SkillOutput(
                result=grid.copy(),
                confidence=0.0,
                reasoning="No objects to isolate"
            )
        
        # For simplicity, extract largest object
        # In full implementation, would create multiple outputs
        return self._extract_largest(grid)
    
    def can_apply(self, task: Task) -> float:
        """
        Determine if object cognition is relevant for this task.
        
        Heuristics:
        - Are there discrete objects in the input?
        - Does the output appear to be derived from input objects?
        - Are object properties (count, size, color) important?
        """
        if not task.train_pairs:
            return 0.0
        
        scores = []
        
        for input_grid, output_grid in task.train_pairs:
            # Analyze input objects
            input_analysis = self.analyze_objects(input_grid)
            
            # High relevance if multiple distinct objects
            if input_analysis.num_objects > 1:
                scores.append(0.8)
            elif input_analysis.num_objects == 1:
                scores.append(0.5)
            else:
                scores.append(0.1)
            
            # Check if output relates to objects
            # (e.g., same number of objects, extracted object, etc.)
            output_analysis = self.analyze_objects(output_grid)
            
            # If output has objects with same colors as input, likely relevant
            color_overlap = len(input_analysis.colors_present & output_analysis.colors_present)
            if color_overlap > 0:
                scores.append(0.7)
            
            # If output is smaller (object extraction)
            if (output_grid.shape[0] < input_grid.shape[0] or 
                output_grid.shape[1] < input_grid.shape[1]):
                scores.append(0.6)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def set_connectivity(self, connectivity: int):
        """Set connectivity mode (4 or 8)."""
        if connectivity not in [4, 8]:
            raise ValueError("Connectivity must be 4 or 8")
        self.connectivity = connectivity


# Convenience function to create and use the skill
def detect_objects(grid: Grid, connectivity: int = 4) -> ObjectAnalysis:
    """Quick function to detect objects in a grid."""
    skill = ObjectCognitionSkill()
    skill.set_connectivity(connectivity)
    return skill.analyze_objects(grid)
