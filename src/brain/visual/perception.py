"""
Phase 3 Perception Module - Minimal Viable Pipeline

Combines PositionEncoder, EdgeDetector, RegionDetector, BackgroundDetector
into a single efficient module. Follows CEREBRUM architecture critique:
- Single hypothesis (CC segmentation only)
- Single config (loose defaults)
- Deterministic, 100% accuracy on basic perception
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional
import numpy as np
from scipy.ndimage import label


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Object:
    """
    A detected object in the grid.
    Minimal representation for Phase 3.
    """
    object_id: int
    color: int
    cells: Set[Tuple[int, int]]  # Set of (row, col) positions
    
    # Computed lazily
    _bbox: Optional[Tuple[int, int, int, int]] = field(default=None, repr=False)
    _centroid: Optional[Tuple[float, float]] = field(default=None, repr=False)
    
    @property
    def size(self) -> int:
        """Number of cells in this object."""
        return len(self.cells)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box as (min_row, min_col, max_row, max_col)."""
        if self._bbox is None:
            rows = [r for r, c in self.cells]
            cols = [c for r, c in self.cells]
            self._bbox = (min(rows), min(cols), max(rows), max(cols))
        return self._bbox
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Center of mass (row, col)."""
        if self._centroid is None:
            rows = [r for r, c in self.cells]
            cols = [c for r, c in self.cells]
            self._centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
        return self._centroid
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bbox[3] - self.bbox[1] + 1
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class VisualOutput:
    """
    Output of the Visual Cortex for Phase 3.
    Minimal: just objects and background.
    """
    grid: np.ndarray                    # Original grid
    background_color: int               # Detected background
    objects: List[Object]               # Detected foreground objects
    
    @property
    def height(self) -> int:
        return self.grid.shape[0]
    
    @property
    def width(self) -> int:
        return self.grid.shape[1]
    
    @property
    def num_objects(self) -> int:
        return len(self.objects)
    
    def get_object_at(self, row: int, col: int) -> Optional[Object]:
        """Get object containing position (row, col)."""
        for obj in self.objects:
            if (row, col) in obj.cells:
                return obj
        return None


# ============================================================================
# PHASE 3 VISUAL CORTEX (Minimal Viable)
# ============================================================================

class VisualCortex:
    """
    Phase 3 Visual Cortex - Minimal Viable Pipeline.
    
    Combines all perception in one efficient pass:
    1. Detect background (most frequent border color)
    2. Find connected components (foreground objects)
    
    No multi-hypothesis, no stratification, no adaptation.
    """
    
    def __init__(self):
        # Connectivity: 4-connected (only horizontal/vertical neighbors)
        # This is the ARC standard interpretation
        self._structure = np.array([[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]])
    
    def process(self, grid: np.ndarray) -> VisualOutput:
        """
        Process a grid and extract visual features.
        
        Args:
            grid: 2D numpy array of color values (0-9)
            
        Returns:
            VisualOutput with detected background and objects
        """
        grid = np.asarray(grid, dtype=np.int8)
        
        # Step 1: Detect background (O(H*W) - single pass)
        background = self._detect_background(grid)
        
        # Step 2: Find objects via connected components (O(H*W))
        objects = self._find_objects(grid, background)
        
        return VisualOutput(
            grid=grid,
            background_color=background,
            objects=objects
        )
    
    def _detect_background(self, grid: np.ndarray) -> int:
        """
        Detect background color.
        
        Phase 3 strategy: Most frequent color on the border.
        Falls back to most frequent color overall if border is uniform.
        This is O(perimeter) ≈ O(H+W).
        """
        H, W = grid.shape
        
        # Collect border colors (more efficient than creating border mask)
        border_colors = []
        border_colors.extend(grid[0, :].tolist())      # Top row
        border_colors.extend(grid[-1, :].tolist())     # Bottom row
        border_colors.extend(grid[1:-1, 0].tolist())   # Left column (excl corners)
        border_colors.extend(grid[1:-1, -1].tolist())  # Right column (excl corners)
        
        if not border_colors:
            # Tiny grid (1x1), use the only color
            return int(grid[0, 0])
        
        # Count border colors
        border_counts = np.bincount(border_colors, minlength=10)
        most_frequent_border = int(np.argmax(border_counts))
        
        # If border is >50% one color, use it
        if border_counts[most_frequent_border] > len(border_colors) * 0.5:
            return most_frequent_border
        
        # Fallback: most frequent overall
        overall_counts = np.bincount(grid.flatten(), minlength=10)
        return int(np.argmax(overall_counts))
    
    def _find_objects(self, grid: np.ndarray, background: int) -> List[Object]:
        """
        Find foreground objects via connected components.
        
        Phase 3: Single segmentation strategy (CC by color).
        Uses scipy.ndimage.label for efficiency.
        """
        objects = []
        object_id = 0
        
        # Process each foreground color
        unique_colors = np.unique(grid)
        
        for color in unique_colors:
            if color == background:
                continue
            
            # Binary mask for this color
            mask = (grid == color)
            
            # Find connected components
            labeled, num_components = label(mask, structure=self._structure)
            
            # Extract each component
            for component_id in range(1, num_components + 1):
                # Find all cells in this component
                positions = np.argwhere(labeled == component_id)
                cells = {(int(r), int(c)) for r, c in positions}
                
                objects.append(Object(
                    object_id=object_id,
                    color=int(color),
                    cells=cells
                ))
                object_id += 1
        
        return objects


# ============================================================================
# POSITION ENCODING (for transformation detection)
# ============================================================================

def compute_displacement(obj_in: Object, obj_out: Object) -> Tuple[int, int]:
    """
    Compute displacement vector between two objects.
    Uses centroid for robustness.
    
    Returns (row_delta, col_delta).
    """
    c_in = obj_in.centroid
    c_out = obj_out.centroid
    return (round(c_out[0] - c_in[0]), round(c_out[1] - c_in[1]))


def objects_same_shape(obj1: Object, obj2: Object) -> bool:
    """
    Check if two objects have the same shape (ignoring position).
    
    Compares the normalized cell positions.
    """
    if obj1.size != obj2.size:
        return False
    
    # Normalize to top-left corner
    r1_min, c1_min, _, _ = obj1.bbox
    r2_min, c2_min, _, _ = obj2.bbox
    
    norm1 = {(r - r1_min, c - c1_min) for r, c in obj1.cells}
    norm2 = {(r - r2_min, c - c2_min) for r, c in obj2.cells}
    
    return norm1 == norm2


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_perception():
    """
    Quick verification that perception works correctly.
    """
    print("=" * 60)
    print("PHASE 3 PERCEPTION VERIFICATION")
    print("=" * 60)
    
    vc = VisualCortex()
    
    # Test 1: Simple grid with black background
    print("\n1. Simple grid (black background):")
    grid1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0],
    ])
    result1 = vc.process(grid1)
    print(f"   Background: {result1.background_color} (expected: 0)")
    print(f"   Objects: {result1.num_objects} (expected: 2)")
    for obj in result1.objects:
        print(f"     - Color {obj.color}, size {obj.size}, centroid {obj.centroid}")
    
    # Test 2: Non-black background
    print("\n2. Blue background grid:")
    grid2 = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1],
    ])
    result2 = vc.process(grid2)
    print(f"   Background: {result2.background_color} (expected: 1)")
    print(f"   Objects: {result2.num_objects} (expected: 1)")
    
    # Test 3: Multiple disconnected objects
    print("\n3. Multiple disconnected objects:")
    grid3 = np.array([
        [0, 1, 0, 2, 0],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 0, 4],
    ])
    result3 = vc.process(grid3)
    print(f"   Objects: {result3.num_objects} (expected: 4)")
    
    # Test 4: Shape comparison
    print("\n4. Shape comparison:")
    obj_a = Object(0, 1, {(0, 0), (0, 1), (1, 0), (1, 1)})  # 2x2 square at origin
    obj_b = Object(1, 1, {(5, 5), (5, 6), (6, 5), (6, 6)})  # 2x2 square elsewhere
    obj_c = Object(2, 1, {(0, 0), (0, 1), (0, 2)})          # 1x3 line
    print(f"   Square == Square (different pos): {objects_same_shape(obj_a, obj_b)} (expected: True)")
    print(f"   Square == Line: {objects_same_shape(obj_a, obj_c)} (expected: False)")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    verify_perception()
