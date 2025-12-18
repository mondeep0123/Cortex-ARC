"""Brute force solver - tries all simple transformations."""

from __future__ import annotations
from typing import List, Optional, Callable
from itertools import product

from .base import Solver, SolverResult
from ..core.grid import Grid
from ..core.task import Task, TaskPair
from ..core.transforms import TransformLibrary


class BruteForceSolver(Solver):
    """
    Baseline solver that tries simple transformations.
    
    This is a good sanity check and can solve some easy tasks.
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        name: str = "brute_force"
    ):
        super().__init__(name=name, max_depth=max_depth)
        self.max_depth = max_depth
        self.transforms = TransformLibrary()
        
        # Define simple transform functions
        self.simple_transforms = [
            ("identity", lambda g: g.copy()),
            ("rotate_90", lambda g: g.rotate_90()),
            ("rotate_180", lambda g: g.rotate_180()),
            ("rotate_270", lambda g: g.rotate_270()),
            ("flip_h", lambda g: g.flip_horizontal()),
            ("flip_v", lambda g: g.flip_vertical()),
            ("transpose", lambda g: g.transpose()),
        ]
        
        # Add color replacements for common cases
        for old_color in range(10):
            for new_color in range(10):
                if old_color != new_color:
                    self.simple_transforms.append(
                        (f"replace_{old_color}_{new_color}",
                         lambda g, o=old_color, n=new_color: g.replace_color(o, n))
                    )
    
    def _transform_matches(
        self,
        transform_fn: Callable[[Grid], Grid],
        pairs: List[TaskPair]
    ) -> bool:
        """Check if a transform works for all training pairs."""
        for pair in pairs:
            try:
                result = transform_fn(pair.input)
                if result != pair.output:
                    return False
            except Exception:
                return False
        return True
    
    def _find_matching_transform(
        self,
        task: Task,
        depth: int = 1
    ) -> Optional[Callable[[Grid], Grid]]:
        """
        Find a transform that works for all training pairs.
        
        Args:
            task: Task to solve
            depth: How many transforms to chain
        
        Returns:
            Transform function if found, None otherwise
        """
        if depth == 1:
            # Try single transforms
            for name, transform in self.simple_transforms:
                if self._transform_matches(transform, task.train):
                    return transform
        
        elif depth == 2:
            # Try pairs of transforms
            for (name1, t1), (name2, t2) in product(
                self.simple_transforms[:7],  # Only geometric for depth 2
                self.simple_transforms[:7]
            ):
                def combined(g, t1=t1, t2=t2):
                    return t2(t1(g))
                
                if self._transform_matches(combined, task.train):
                    return combined
        
        elif depth >= 3:
            # Try triples (limited search)
            geometric = self.simple_transforms[:7]
            for (_, t1), (_, t2), (_, t3) in product(geometric, geometric, geometric):
                def combined(g, t1=t1, t2=t2, t3=t3):
                    return t3(t2(t1(g)))
                
                if self._transform_matches(combined, task.train):
                    return combined
        
        return None
    
    def _try_output_shape_heuristics(self, task: Task) -> Optional[Callable[[Grid], Grid]]:
        """
        Try heuristics based on output shape patterns.
        """
        # Check if output is always same size as input
        if all(p.input.shape == p.output.shape for p in task.train):
            # Same shape - geometric or color transforms only
            pass
        
        # Check if output is cropped content
        for pair in task.train:
            cropped = pair.input.crop_to_content()
            if cropped is not None and cropped == pair.output:
                return lambda g: g.crop_to_content() or g
        
        # Check if output is scaled version
        for scale in [2, 3, 4]:
            matches = True
            for pair in task.train:
                if pair.input.scale(scale) != pair.output:
                    matches = False
                    break
            if matches:
                return lambda g, s=scale: g.scale(s)
        
        return None
    
    def solve(self, task: Task) -> SolverResult:
        """Solve the task using brute force search."""
        
        # Try output shape heuristics first
        transform = self._try_output_shape_heuristics(task)
        
        # Try brute force search at increasing depths
        if transform is None:
            for depth in range(1, self.max_depth + 1):
                transform = self._find_matching_transform(task, depth=depth)
                if transform is not None:
                    break
        
        # Generate predictions
        predictions = []
        confidence = []
        
        for test_pair in task.test:
            if transform is not None:
                try:
                    pred = transform(test_pair.input)
                    predictions.append([pred])
                    confidence.append([0.9])  # High confidence if we found a match
                except Exception:
                    predictions.append([test_pair.input.copy()])
                    confidence.append([0.1])
            else:
                # No transform found - return input as fallback
                predictions.append([test_pair.input.copy()])
                confidence.append([0.1])
        
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            confidence=confidence,
            metadata={"found_transform": transform is not None}
        )


class PatternMatchingSolver(Solver):
    """
    Solver that looks for specific patterns in inputs.
    """
    
    def __init__(self, name: str = "pattern_matching"):
        super().__init__(name=name)
    
    def _find_pattern_positions(
        self,
        grid: Grid,
        pattern: Grid
    ) -> List[tuple]:
        """Find all positions where pattern appears in grid."""
        positions = []
        ph, pw = pattern.shape
        
        for row in range(grid.height - ph + 1):
            for col in range(grid.width - pw + 1):
                region = grid.data[row:row+ph, col:col+pw]
                if (region == pattern.data).all():
                    positions.append((row, col))
        
        return positions
    
    def solve(self, task: Task) -> SolverResult:
        """Solve using pattern matching."""
        # This is a simplified implementation
        # A full implementation would:
        # 1. Find recurring patterns in training examples
        # 2. Learn transformation rules for patterns
        # 3. Apply to test cases
        
        predictions = []
        confidence = []
        
        for test_pair in task.test:
            # Fallback: return input
            predictions.append([test_pair.input.copy()])
            confidence.append([0.1])
        
        return SolverResult(
            task_id=task.task_id,
            predictions=predictions,
            confidence=confidence
        )
