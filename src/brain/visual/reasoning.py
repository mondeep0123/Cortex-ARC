"""
Phase 3 Reasoning Module - Object Matching and Transformation Detection

Implements:
- ObjectMatcher: Hungarian algorithm for 1-to-1 correspondence
- ComparisonModule: Detect transformation signatures
- TransformationRule: Represents detected rules

Follows CEREBRUM architecture:
- Single config (loose defaults)
- Global analysis only (no stratification)
- Simple transformations only (translate, rotate, recolor)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum, auto
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from .perception import Object, VisualOutput, compute_displacement, objects_same_shape
except ImportError:
    from perception import Object, VisualOutput, compute_displacement, objects_same_shape


# ============================================================================
# TRANSFORMATION TYPES
# ============================================================================

class TransformType(Enum):
    """Types of transformations we can detect."""
    # Basic transforms (Phase 3)
    IDENTITY = auto()       # No change
    TRANSLATE = auto()      # Move objects
    ROTATE_90 = auto()      # 90° clockwise
    ROTATE_180 = auto()     # 180°
    ROTATE_270 = auto()     # 270° clockwise
    FLIP_H = auto()         # Horizontal flip
    FLIP_V = auto()         # Vertical flip
    RECOLOR = auto()        # Change colors
    DELETE = auto()         # Object removed
    CREATE = auto()         # Object created
    
    # Extended transforms (Phase 3.5)
    SCALE_2X = auto()       # Scale up 2x
    SCALE_3X = auto()       # Scale up 3x
    SCALE_DOWN_2X = auto()  # Scale down 2x
    CROP = auto()           # Crop to content
    TILE_2X = auto()        # Tile 2x2
    TILE_3X = auto()        # Tile 3x3
    GRAVITY_DOWN = auto()   # Objects fall down
    GRAVITY_LEFT = auto()   # Objects move left
    TRANSPOSE = auto()      # Swap rows and columns
    
    UNKNOWN = auto()        # Can't determine


# ============================================================================
# MATCHER CONFIGURATION
# ============================================================================

@dataclass
class MatcherConfig:
    """
    Phase 3 matcher configuration.
    LOOSE defaults per CEREBRUM critique.
    """
    # Cost weights (loose = lower penalties)
    color_mismatch: float = 5.0         # Penalty for color difference
    size_mismatch_weight: float = 0.5   # Weight for size difference
    position_weight: float = 0.1        # Weight for position distance
    shape_mismatch: float = 8.0         # Penalty for different shapes
    
    # Threshold for "unmatched" penalty
    unmatched_penalty: float = 20.0     # Cost of deletion/creation


# ============================================================================
# OBJECT CORRESPONDENCE
# ============================================================================

@dataclass
class ObjectMatch:
    """A single matched pair of objects."""
    input_id: int
    output_id: int
    cost: float
    
@dataclass
class ObjectCorrespondence:
    """
    Result of matching objects between input and output grids.
    """
    matches: List[ObjectMatch]          # Matched pairs
    unmatched_input: List[int]          # Object IDs deleted
    unmatched_output: List[int]         # Object IDs created
    total_cost: float
    
    @property
    def num_matches(self) -> int:
        return len(self.matches)
    
    def get_match_for_input(self, input_id: int) -> Optional[ObjectMatch]:
        """Get the match for a given input object."""
        for m in self.matches:
            if m.input_id == input_id:
                return m
        return None


# ============================================================================
# OBJECT MATCHER (Hungarian Algorithm)
# ============================================================================

class ObjectMatcher:
    """
    Matches objects between input and output using Hungarian algorithm.
    
    Phase 3: Single-pass, single config, no relaxation.
    """
    
    def __init__(self, config: Optional[MatcherConfig] = None):
        self.config = config or MatcherConfig()
    
    def match(
        self,
        input_objects: List[Object],
        output_objects: List[Object]
    ) -> ObjectCorrespondence:
        """
        Find optimal 1-to-1 correspondence between objects.
        
        Uses Hungarian algorithm for minimum cost assignment.
        """
        n_in = len(input_objects)
        n_out = len(output_objects)
        
        # Edge case: no objects
        if n_in == 0 and n_out == 0:
            return ObjectCorrespondence([], [], [], 0.0)
        
        if n_in == 0:
            return ObjectCorrespondence(
                matches=[],
                unmatched_input=[],
                unmatched_output=[o.object_id for o in output_objects],
                total_cost=n_out * self.config.unmatched_penalty
            )
        
        if n_out == 0:
            return ObjectCorrespondence(
                matches=[],
                unmatched_input=[o.object_id for o in input_objects],
                unmatched_output=[],
                total_cost=n_in * self.config.unmatched_penalty
            )
        
        # Build cost matrix
        # Rows = input objects, Cols = output objects
        # Pad to square matrix for Hungarian algorithm
        max_dim = max(n_in, n_out)
        cost_matrix = np.full((max_dim, max_dim), self.config.unmatched_penalty)
        
        for i, obj_in in enumerate(input_objects):
            for j, obj_out in enumerate(output_objects):
                cost_matrix[i, j] = self._compute_cost(obj_in, obj_out)
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matches
        matches = []
        unmatched_in = []
        unmatched_out = set(range(n_out))
        total_cost = 0.0
        
        for i, j in zip(row_ind, col_ind):
            if i < n_in and j < n_out:
                # Real match
                cost = cost_matrix[i, j]
                if cost < self.config.unmatched_penalty:
                    matches.append(ObjectMatch(
                        input_id=input_objects[i].object_id,
                        output_id=output_objects[j].object_id,
                        cost=cost
                    ))
                    unmatched_out.discard(j)
                    total_cost += cost
                else:
                    # Cost too high, treat as unmatched
                    unmatched_in.append(input_objects[i].object_id)
                    total_cost += self.config.unmatched_penalty
            elif i < n_in:
                # Input object unmatched (deleted)
                unmatched_in.append(input_objects[i].object_id)
                total_cost += self.config.unmatched_penalty
        
        unmatched_out_ids = [output_objects[j].object_id for j in unmatched_out]
        total_cost += len(unmatched_out_ids) * self.config.unmatched_penalty
        
        return ObjectCorrespondence(
            matches=matches,
            unmatched_input=unmatched_in,
            unmatched_output=unmatched_out_ids,
            total_cost=total_cost
        )
    
    def _compute_cost(self, obj_in: Object, obj_out: Object) -> float:
        """
        Compute matching cost between two objects.
        Lower cost = better match.
        """
        cost = 0.0
        
        # Color mismatch
        if obj_in.color != obj_out.color:
            cost += self.config.color_mismatch
        
        # Size difference (normalized)
        size_diff = abs(obj_in.size - obj_out.size) / max(obj_in.size, obj_out.size, 1)
        cost += size_diff * self.config.size_mismatch_weight * 10
        
        # Position distance (Euclidean between centroids)
        c_in = obj_in.centroid
        c_out = obj_out.centroid
        dist = ((c_in[0] - c_out[0])**2 + (c_in[1] - c_out[1])**2) ** 0.5
        cost += dist * self.config.position_weight
        
        # Shape mismatch
        if not objects_same_shape(obj_in, obj_out):
            cost += self.config.shape_mismatch
        
        return cost


# ============================================================================
# TRANSFORMATION SIGNATURE
# ============================================================================

@dataclass
class TransformationSignature:
    """
    Phase 3 transformation signature.
    Describes what changed between input and output.
    """
    # Global flags
    has_translation: bool = False
    has_rotation: bool = False
    has_recolor: bool = False
    has_deletion: bool = False
    has_creation: bool = False
    
    # Translation info
    translation_vector: Optional[Tuple[int, int]] = None
    all_same_displacement: bool = False
    
    # Rotation info
    rotation_type: Optional[TransformType] = None
    
    # Color mapping
    color_mapping: Dict[int, int] = field(default_factory=dict)
    
    # Deleted/Created
    deleted_ids: List[int] = field(default_factory=list)
    created_ids: List[int] = field(default_factory=list)
    
    # Quality metrics
    num_matched: int = 0
    num_transformed: int = 0
    
    @property
    def is_identity(self) -> bool:
        """Check if transformation is identity (no change)."""
        return (not self.has_translation and 
                not self.has_rotation and 
                not self.has_recolor and 
                not self.has_deletion and 
                not self.has_creation)
    
    @property
    def primary_transform(self) -> TransformType:
        """Get the primary transformation type."""
        if self.is_identity:
            return TransformType.IDENTITY
        if self.has_translation:
            return TransformType.TRANSLATE
        if self.has_rotation:
            return self.rotation_type or TransformType.UNKNOWN
        if self.has_recolor:
            return TransformType.RECOLOR
        if self.has_deletion:
            return TransformType.DELETE
        if self.has_creation:
            return TransformType.CREATE
        return TransformType.UNKNOWN


# ============================================================================
# GRID-LEVEL TRANSFORMATION DETECTION
# ============================================================================

def detect_grid_transform(input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[TransformType]:
    """
    Detect grid-level transformations (rotation, flip).
    Checks if the entire grid was transformed.
    
    Returns the TransformType if detected, None otherwise.
    
    NOTE: We check ROTATIONS before FLIPS because:
    1. Rotations are more common in ARC
    2. For symmetric grids, rotation and flip can produce same result
    
    Priority order: rot90 > rot180 > rot270 > flip_h > flip_v
    """
    H_in, W_in = input_grid.shape
    H_out, W_out = output_grid.shape
    
    # Identity check
    if np.array_equal(input_grid, output_grid):
        return None  # No transform
    
    # Check 90° clockwise (works for any grid if dimensions swap)
    rot90 = np.rot90(input_grid, k=-1)
    if rot90.shape == output_grid.shape and np.array_equal(rot90, output_grid):
        return TransformType.ROTATE_90
    
    # Check 180° rotation (only same-size grids)
    if H_in == H_out and W_in == W_out:
        rot180 = np.rot90(input_grid, k=2)
        if np.array_equal(rot180, output_grid):
            return TransformType.ROTATE_180
    
    # Check 270° clockwise (works for any grid if dimensions swap)
    rot270 = np.rot90(input_grid, k=1)
    if rot270.shape == output_grid.shape and np.array_equal(rot270, output_grid):
        return TransformType.ROTATE_270
    
    # Now check flips (only same-size grids)
    if H_in == H_out and W_in == W_out:
        # Check horizontal flip
        flip_h = np.fliplr(input_grid)
        if np.array_equal(flip_h, output_grid):
            return TransformType.FLIP_H
        
        # Check vertical flip
        flip_v = np.flipud(input_grid)
        if np.array_equal(flip_v, output_grid):
            return TransformType.FLIP_V
        
        # Check transpose
        if H_in == W_in:  # Square grid
            transposed = input_grid.T
            if np.array_equal(transposed, output_grid):
                return TransformType.TRANSPOSE
    
    # Check scaling (output is multiple of input)
    if H_out == H_in * 2 and W_out == W_in * 2:
        scaled_2x = np.repeat(np.repeat(input_grid, 2, axis=0), 2, axis=1)
        if np.array_equal(scaled_2x, output_grid):
            return TransformType.SCALE_2X
    
    if H_out == H_in * 3 and W_out == W_in * 3:
        scaled_3x = np.repeat(np.repeat(input_grid, 3, axis=0), 3, axis=1)
        if np.array_equal(scaled_3x, output_grid):
            return TransformType.SCALE_3X
    
    # Check scale down (input is multiple of output)
    if H_in == H_out * 2 and W_in == W_out * 2:
        # Check if each 2x2 block maps to a single cell
        scaled_down = input_grid[::2, ::2]
        if np.array_equal(scaled_down, output_grid):
            return TransformType.SCALE_DOWN_2X
    
    # Check tiling (output is input repeated)
    if H_out == H_in * 2 and W_out == W_in * 2:
        tiled = np.tile(input_grid, (2, 2))
        if np.array_equal(tiled, output_grid):
            return TransformType.TILE_2X
    
    if H_out == H_in * 3 and W_out == W_in * 3:
        tiled = np.tile(input_grid, (3, 3))
        if np.array_equal(tiled, output_grid):
            return TransformType.TILE_3X
    
    # Check crop (output is a subgrid of input)
    if H_out <= H_in and W_out <= W_in and (H_out < H_in or W_out < W_in):
        # Try to find output as a subgrid
        for r in range(H_in - H_out + 1):
            for c in range(W_in - W_out + 1):
                subgrid = input_grid[r:r+H_out, c:c+W_out]
                if np.array_equal(subgrid, output_grid):
                    return TransformType.CROP
    
    return None


# ============================================================================
# COMPARISON MODULE
# ============================================================================

class ComparisonModule:
    """
    Analyzes matched objects to detect transformation patterns.
    
    Phase 3: Global analysis only, no stratification.
    Now also detects grid-level transformations (rotation, flip).
    """
    
    def compare(
        self,
        input_visual: VisualOutput,
        output_visual: VisualOutput,
        correspondence: ObjectCorrespondence
    ) -> TransformationSignature:
        """
        Analyze transformation between input and output.
        """
        sig = TransformationSignature()
        
        # Check for grid-level transformations (but don't return early)
        grid_transform = detect_grid_transform(input_visual.grid, output_visual.grid)
        if grid_transform is not None:
            sig.has_rotation = True
            sig.rotation_type = grid_transform
            # NOTE: Don't return early anymore - let extract_rules verify consistency
        
        # Index objects for quick lookup
        input_objs = {o.object_id: o for o in input_visual.objects}
        output_objs = {o.object_id: o for o in output_visual.objects}
        
        sig.num_matched = correspondence.num_matches
        
        # Check deletions/creations
        if correspondence.unmatched_input:
            sig.has_deletion = True
            sig.deleted_ids = correspondence.unmatched_input
        
        if correspondence.unmatched_output:
            sig.has_creation = True
            sig.created_ids = correspondence.unmatched_output
        
        # Analyze matched pairs
        displacements = []
        color_changes = {}
        
        for match in correspondence.matches:
            obj_in = input_objs[match.input_id]
            obj_out = output_objs[match.output_id]
            
            # Check color change
            if obj_in.color != obj_out.color:
                sig.has_recolor = True
                color_changes[obj_in.color] = obj_out.color
            
            # Check displacement
            disp = compute_displacement(obj_in, obj_out)
            if disp != (0, 0):
                sig.has_translation = True
                displacements.append(disp)
        
        # Analyze displacements
        if displacements:
            # Check if all same
            if len(set(displacements)) == 1:
                sig.all_same_displacement = True
                sig.translation_vector = displacements[0]
            else:
                # Use most common
                from collections import Counter
                most_common = Counter(displacements).most_common(1)[0][0]
                sig.translation_vector = most_common
        
        # Record color mapping
        sig.color_mapping = color_changes
        sig.num_transformed = len([d for d in displacements if d != (0, 0)])
        
        return sig


# ============================================================================
# TRANSFORMATION RULE
# ============================================================================

@dataclass
class TransformationRule:
    """
    A detected transformation rule.
    Phase 3: Simple rules only.
    """
    transform_type: TransformType
    parameters: Dict[str, any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def apply(self, grid: np.ndarray) -> np.ndarray:
        """
        Apply this transformation to a grid.
        Returns the transformed grid.
        """
        result = grid.copy()
        
        if self.transform_type == TransformType.IDENTITY:
            return result
        
        elif self.transform_type == TransformType.TRANSLATE:
            dr, dc = self.parameters.get("vector", (0, 0))
            H, W = result.shape
            new_grid = np.zeros_like(result)
            
            for r in range(H):
                for c in range(W):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if result[r, c] != 0:  # Only move non-background
                            new_grid[nr, nc] = result[r, c]
                    # Keep background
                    elif result[r, c] == 0:
                        pass
            return new_grid
        
        elif self.transform_type == TransformType.ROTATE_90:
            return np.rot90(result, k=-1)
        
        elif self.transform_type == TransformType.ROTATE_180:
            return np.rot90(result, k=2)
        
        elif self.transform_type == TransformType.ROTATE_270:
            return np.rot90(result, k=1)
        
        elif self.transform_type == TransformType.FLIP_H:
            return np.fliplr(result)
        
        elif self.transform_type == TransformType.FLIP_V:
            return np.flipud(result)
        
        elif self.transform_type == TransformType.RECOLOR:
            mapping = self.parameters.get("mapping", {})
            for old_color, new_color in mapping.items():
                result[grid == old_color] = new_color
            return result
        
        # Extended transforms
        elif self.transform_type == TransformType.SCALE_2X:
            return np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
        
        elif self.transform_type == TransformType.SCALE_3X:
            return np.repeat(np.repeat(result, 3, axis=0), 3, axis=1)
        
        elif self.transform_type == TransformType.SCALE_DOWN_2X:
            return result[::2, ::2]
        
        elif self.transform_type == TransformType.TILE_2X:
            return np.tile(result, (2, 2))
        
        elif self.transform_type == TransformType.TILE_3X:
            return np.tile(result, (3, 3))
        
        elif self.transform_type == TransformType.TRANSPOSE:
            return result.T
        
        elif self.transform_type == TransformType.CROP:
            # Crop to bounding box of non-zero content
            bg_color = self.parameters.get("bg_color", 0)
            non_bg = np.argwhere(result != bg_color)
            if len(non_bg) == 0:
                return result
            min_r, min_c = non_bg.min(axis=0)
            max_r, max_c = non_bg.max(axis=0)
            return result[min_r:max_r+1, min_c:max_c+1]
        
        elif self.transform_type == TransformType.GRAVITY_DOWN:
            # Move all non-zero cells down as far as possible
            H, W = result.shape
            new_grid = np.zeros_like(result)
            for c in range(W):
                col = result[:, c]
                non_zero = col[col != 0]
                new_grid[H-len(non_zero):, c] = non_zero
            return new_grid
        
        elif self.transform_type == TransformType.GRAVITY_LEFT:
            # Move all non-zero cells left as far as possible
            H, W = result.shape
            new_grid = np.zeros_like(result)
            for r in range(H):
                row = result[r, :]
                non_zero = row[row != 0]
                new_grid[r, :len(non_zero)] = non_zero
            return new_grid
        
        return result
    
    def __repr__(self) -> str:
        return f"Rule({self.transform_type.name}, {self.parameters})"


# ============================================================================
# RULE EXTRACTOR
# ============================================================================

def extract_rules(signatures: List[TransformationSignature]) -> List[TransformationRule]:
    """
    Extract consistent rules from multiple training examples.
    
    Phase 3: Find rules that apply to ALL examples.
    """
    if not signatures:
        return []
    
    rules = []
    
    # Check for consistent rotation/flip (grid-level transforms)
    # MUST be detected in ALL examples to be valid
    rotations = [s.rotation_type for s in signatures if s.has_rotation and s.rotation_type]
    if len(rotations) == len(signatures) and len(set(rotations)) == 1:
        # All examples have the same rotation type
        rules.append(TransformationRule(
            transform_type=rotations[0],
            confidence=1.0
        ))
        return rules  # If rotation detected, that's the only rule needed
    
    # Check for consistent translation
    translations = [s.translation_vector for s in signatures if s.has_translation and s.translation_vector]
    if translations and len(set(translations)) == 1:
        rules.append(TransformationRule(
            transform_type=TransformType.TRANSLATE,
            parameters={"vector": translations[0]},
            confidence=1.0
        ))
    
    # Check for consistent recolor
    color_mappings = [s.color_mapping for s in signatures if s.has_recolor and s.color_mapping]
    if color_mappings:
        # Find intersection of all mappings
        common_mapping = color_mappings[0].copy()
        for mapping in color_mappings[1:]:
            common_mapping = {k: v for k, v in common_mapping.items() 
                            if k in mapping and mapping[k] == v}
        
        if common_mapping:
            rules.append(TransformationRule(
                transform_type=TransformType.RECOLOR,
                parameters={"mapping": common_mapping},
                confidence=len(common_mapping) / len(color_mappings[0]) if color_mappings[0] else 0
            ))
    
    # If no rules found, check for identity
    if not rules and all(s.is_identity for s in signatures):
        rules.append(TransformationRule(
            transform_type=TransformType.IDENTITY,
            confidence=1.0
        ))
    
    return rules


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_reasoning():
    """Verify reasoning module works."""
    try:
        from .perception import VisualCortex
    except ImportError:
        from perception import VisualCortex
    
    print("=" * 60)
    print("PHASE 3 REASONING VERIFICATION")
    print("=" * 60)
    
    vc = VisualCortex()
    matcher = ObjectMatcher()
    comparison = ComparisonModule()
    
    # Test 1: Translation detection
    print("\n1. Translation detection:")
    grid_in = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    grid_out = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    
    vis_in = vc.process(grid_in)
    vis_out = vc.process(grid_out)
    corr = matcher.match(vis_in.objects, vis_out.objects)
    sig = comparison.compare(vis_in, vis_out, corr)
    
    print(f"   Translation: {sig.has_translation} (expected: True)")
    print(f"   Vector: {sig.translation_vector} (expected: (0, 1))")
    print(f"   All same: {sig.all_same_displacement}")
    
    # Test 2: Recolor detection
    print("\n2. Recolor detection:")
    grid_in2 = np.array([
        [0, 1, 0],
        [0, 1, 0],
    ])
    grid_out2 = np.array([
        [0, 2, 0],
        [0, 2, 0],
    ])
    
    vis_in2 = vc.process(grid_in2)
    vis_out2 = vc.process(grid_out2)
    corr2 = matcher.match(vis_in2.objects, vis_out2.objects)
    sig2 = comparison.compare(vis_in2, vis_out2, corr2)
    
    print(f"   Recolor: {sig2.has_recolor} (expected: True)")
    print(f"   Mapping: {sig2.color_mapping} (expected: {{1: 2}})")
    
    # Test 3: Rule extraction
    print("\n3. Rule extraction:")
    rules = extract_rules([sig])
    print(f"   Rules: {rules}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_reasoning()
